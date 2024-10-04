//
// Created by ben on 25.09.24.
//

#include "../include/plane_arap.h"


#include <cassert>

#include <iostream>

void removeRow(Eigen::MatrixXd &matrix, unsigned int rowToRemove) {
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (rowToRemove < numRows)
        matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(
            rowToRemove + 1, 0, numRows - rowToRemove, numCols);

    matrix.conservativeResize(numRows, numCols);
}

void removeColumn(Eigen::MatrixXd &matrix, unsigned int colToRemove) {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (colToRemove < numCols)
        matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(
            0, colToRemove + 1, numRows, numCols - colToRemove);

    matrix.conservativeResize(numRows, numCols);
}

template<typename Scalar>
using MatrixXX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

bool plane_arap_precomputation(
    poly_mesh_data &mesh_data,
    plane_arap_data &data,
    const Eigen::VectorXi &b) {
    using namespace std;
    using namespace Eigen;
    // number of vertices
    const int n = mesh_data.V.rows();

    int mapTo = 0;
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        bool inB = false;
        int bIndex = 0;
        for (int j = 0; j < b.size(); j++) {
            if (b(j) == i) {
                inB = true;
                bIndex = j;
                break;
            }
        }
        if (inB) {
            data.positions.push_back(-bIndex - 1);
            data.positions.push_back(-bIndex - 1);
            data.positions.push_back(-bIndex - 1);
        } else {
            data.positions.push_back(mapTo);
            mapTo++;
            data.positions.push_back(mapTo);
            mapTo++;
            data.positions.push_back(mapTo);
            mapTo++;
        }
    }

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * mesh_data.V.rows(),
                                              3 * mesh_data.V.rows());
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        std::vector val = mesh_data.Hoods[i];
        int size = val.size();
        int v = i * 3;
        L(v, v) = size;
        L(v + 1, v + 1) = size;
        L(v + 2, v + 2) = size;
        for (auto j: val) {
            //int v2 = data.positions[j * 3];
            int v2 = j * 3;
            if (v2 < 0) {
                continue;
            }
            L(v, v2) = -1;
            L(v + 1, v2 + 1) = -1;
            L(v + 2, v2 + 2) = -1;
        }
    }

    data.L = L;
    data.Polygons = mesh_data.Polygons;
    data.b = b;
    return true;
}


bool plane_arap_solve(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    using namespace Eigen;
    using namespace std;
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const int n = mesh_data.V.rows();
    // Eigen::MatrixXd N = Eigen::MatrixXd::Zero(mesh_data.F.rows(), mesh_data.V.rows() * 3);
    // for (int i = 0; i < mesh_data.F.rows(); i++) {
    //     Eigen::Vector3d normal = mesh_data.Polygons.row(i).head(3).normalized();
    //     int size = faceSize(mesh_data.F.row(i));
    //     for (int j = 0; j < size; j++) {
    //         int vert = mesh_data.F(i, j);
    //         N.block<1, 3>(i, 3 * vert) = normal.transpose() / size;
    //     }
    // }
    //
    // Eigen::VectorXd d(mesh_data.Polygons.rows());
    // for (int i = 0; i < mesh_data.Polygons.rows(); i++) {
    //     d(i) = mesh_data.Polygons(i, 3);
    // }
    std::vector<Eigen::MatrixXd> invNs;
    std::vector<std::vector<int> > nIdx;

    Eigen::VectorXd nV(mesh_data.V.rows() * 3);

    for (int i = 0; i < mesh_data.V.rows(); i++) {
        std::set<int> polygons = mesh_data.VertPolygons[i];
        MatrixXd N(polygons.size(), 3);
        Eigen::VectorXd ds(polygons.size());
        std::vector<int> idx;
        int j = 0;
        for (auto pol: polygons) {
            N.row(j) = mesh_data.Polygons.row(pol).head(3);
            idx.push_back(pol);

            ds(j) = mesh_data.Polygons(pol, 3);
            j++;
        }
        Eigen::MatrixXd NInv = N.completeOrthogonalDecomposition().pseudoInverse();
        nV.segment(i * 3, 3) = NInv * ds;

        invNs.push_back(NInv);
        nIdx.push_back(idx);
    }

    Eigen::MatrixXd NInv = Eigen::MatrixXd::Zero(mesh_data.V.rows() * 3, mesh_data.Polygons.rows());
    for (int i = 0; i < nIdx.size(); i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < nIdx[i].size(); k++) {
                NInv(i * 3 + j, nIdx[i][k]) += invNs[i](j, k);
            }
        }
    }


    // Eigen::VectorXd nV = NInv * d;
    //test


    Eigen::VectorXd d(mesh_data.Polygons.rows());
    for (int i = 0; i < d.size(); i++) {
        d(i) = mesh_data.Polygons(i, 3);
    }

    // std::cout << N << std::endl;
    // std::cout << d << std::endl;
    // std::cout << "verts" << std::endl;
    // std::cout << nV << std::endl;
    // std::cout << "test" << std::endl;
    // std::cout << NInv * d << std::endl;
    // std::cout << "test2" << std::endl;
    // std::cout << N * testV << std::endl;
    //
    //test end

    double theta = 30.0 * M_PI / 180.0;
    Eigen::Matrix3d rotationMatrix;
    rotationMatrix << std::cos(theta), -std::sin(theta), 0,
            std::sin(theta), std::cos(theta), 0,
            0, 0, 1;

    MatrixXd S(mesh_data.V.rows() * 3, 3);


    for (int i = 0; i < mesh_data.V.rows(); i++) {
        Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(mesh_data.Hoods[i].size(), 3);
        Eigen::MatrixXd V2 = Eigen::MatrixXd::Zero(mesh_data.Hoods[i].size(), 3);

        Eigen::Vector3d originalVert = mesh_data.V.row(i);
        Eigen::Vector3d newVert = nV.segment(3 * i, 3);
        int j = 0;
        for (auto v: mesh_data.Hoods[i]) {
            Eigen::Vector3d originalNeighbor = mesh_data.V.row(v);
            Eigen::Vector3d newNeighbor = nV.segment(3 * v, 3);

            V1.row(j) += originalVert - originalNeighbor;
            V2.row(j) += newVert - newNeighbor;

            j++;
        }
        Eigen::Matrix3d s = V1.transpose() * V2;
        // std::cout << V1 << std::endl;
        // std::cout << V2 << std::endl;
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                S(x * mesh_data.V.rows() + i, y) = s(x, y);
            }
        }
    }
    S /= S.array().abs().maxCoeff();


    const int Rdim = 3;
    MatrixXd R(Rdim, 3 * mesh_data.V.rows());
    igl::fit_rotations(S, true, R);

    // std::cout << R << std::endl;

    Eigen::MatrixXd M = data.L * NInv;

    std::vector<int> gons;
    std::vector<double> dists;

    for (int i = 0; i < data.b.size(); i++) {
        Eigen::Vector4d vecs[3];
        int j = 0;
        for (auto k: mesh_data.VertPolygons[data.b(i)]) {
            vecs[j] = mesh_data.Polygons.row(k);
            j++;
        }
        Eigen::Vector3d normal1 = vecs[0].head(3).normalized();
        Eigen::Vector3d normal2 = vecs[1].head(3).normalized();
        Eigen::Vector3d normal3 = vecs[2].head(3).normalized();

        Eigen::Matrix3d m;
        m.row(0) = normal1;
        m.row(1) = normal2;
        m.row(2) = normal3;

        Eigen::Vector3d dist = m * bc.row(i).transpose();
        j = 0;
        for (auto k: mesh_data.VertPolygons[data.b(i)]) {
            gons.push_back(k);
            dists.push_back(dist(j));
            mesh_data.Polygons(k, 3) = dist(j);
            j++;
        }
    }


    Eigen::VectorXd b(3 * mesh_data.V.rows() - 3 * data.b.size());
    int row = 0;
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        if (data.positions[i * 3] < 0) {
            continue;
        }
        Eigen::Matrix3d rot = R.block<3, 3>(0, i * 3);
        Eigen::Vector3d rightSide = Eigen::Vector3d::Zero();

        for (auto v: mesh_data.Hoods[i]) {
            rightSide -= rot * (mesh_data.V.row(v) - mesh_data.V.row(i)).transpose();
        }
        for (int j = 0; j < gons.size(); j++) {
            int deletedPoly = gons[j];
            rightSide(0) -= M(3 * i, deletedPoly) * dists[j];
            rightSide(1) -= M(3 * i + 1, deletedPoly) * dists[j];
            rightSide(2) -= M(3 * i + 2, deletedPoly) * dists[j];
        }
        b(row * 3) = rightSide(0);
        b(row * 3 + 1) = rightSide(1);
        b(row * 3 + 2) = rightSide(2);
        row++;
    }

    Eigen::MatrixXd newM(M.rows() - 3 * data.b.size(), M.cols() - dists.size());
    int y = 0;
    for (int i = 0; i < M.cols(); i++) {
        bool deltedCol = false;
        for (int poly = 0; poly < gons.size(); poly++) {
            if (gons[poly] == i) {
                deltedCol = true;
            }
        }
        if (deltedCol) {
            continue;
        }
        for (int j = 0; j < M.rows(); j++) {
            int x = data.positions[j];
            if (x < 0) {
                continue;
            }
            newM(x, y) = M(j, i);
        }
        y++;
    }

    Eigen::VectorXd bestDistances = newM.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    y = 0;
    for (int i = 0; i < mesh_data.Polygons.rows(); i++) {
        bool skipped = false;
        for (int poly = 0; poly < gons.size(); poly++) {
            if (gons[poly] == i) {
                skipped = true;
            }
        }
        if (skipped) {
            continue;
        }

        mesh_data.Polygons(i, 3) = bestDistances(y);

        y++;
    }

    return true;
}
