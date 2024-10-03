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
    //calculate L matrix
    // Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * mesh_data.V.rows() - b.size() * 3,
    //                                           3 * mesh_data.V.rows() - b.size() * 3);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * mesh_data.V.rows(),
                                              3 * mesh_data.V.rows());
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        // bool inB = false;
        // for (int j = 0; j < b.size(); j++) {
        //     if (b(j) == i) {
        //         inB = true;
        //         break;
        //     }
        // }
        // if (data.positions[i * 3] < 0) {
        //     continue;
        // }
        // if (inB) {
        //     L(key * 3, key * 3) = 1;
        //     L(key * 3 + 1, key * 3 + 1) = 1;
        //     L(key * 3 + 2, key * 3 + 2) = 1;
        //     continue;
        // }
        std::set val = mesh_data.Hoods[i];
        int size = val.size();
        //int v = data.positions[i * 3];
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

    //remove b rols and cols

    // for (int i = 0; i < b.size(); i++) {
    //     int vert = b(i);
    //     L(3 * mesh_data.V.rows() + 3 * i, vert * 3) = 100;
    //     L(3 * mesh_data.V.rows() + 3 * i + 1, vert * 3 + 1) = 100;
    //     L(3 * mesh_data.V.rows() + 3 * i + 2, vert * 3 + 2) = 100;
    // }


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
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(mesh_data.F.rows(), mesh_data.V.rows() * 3);
    for (int i = 0; i < mesh_data.F.rows(); i++) {
        Eigen::Vector3d normal = mesh_data.Polygons.row(i).head(3);
        int size = faceSize(mesh_data.F.row(i));
        for (int j = 0; j < size; j++) {
            int vert = mesh_data.F(i, j);
            N.block<1, 3>(i, 3 * vert) = normal.transpose() / size;
        }
    }

    Eigen::VectorXd d(mesh_data.Polygons.rows());
    for (int i = 0; i < mesh_data.Polygons.rows(); i++) {
        d(i) = mesh_data.Polygons(i, 3);
    }


    Eigen::MatrixXd NInv = N.completeOrthogonalDecomposition().pseudoInverse();

    Eigen::VectorXd nV = NInv * d;

    double theta = 30.0 * M_PI / 180.0; // 1 degree in radians
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
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                S(x * mesh_data.V.rows() + i, y) = s(x, y);
            }
        }
    }

    // Eigen::MatrixXd ogCenters(mesh_data.F.rows(), 3);
    // Eigen::MatrixXd newCenters(mesh_data.F.rows(), 3);
    // //rotations:
    // for (int i = 0; i < mesh_data.F.rows(); i++) {
    //     int size = faceSize(mesh_data.F.row(i));
    //
    //     Eigen::Vector3d ogCenter = Eigen::Vector3d::Zero();
    //     Eigen::Vector3d newCenter = Eigen::Vector3d::Zero();
    //
    //     for (int j = 0; j < size; j++) {
    //         ogCenter += mesh_data.V.row(mesh_data.F(i, j)) / size;
    //         Eigen::Vector3d newVert = nV.segment(mesh_data.F(i, j) * 3, 3);
    //         newCenter += newVert / size;
    //     }
    //     ogCenters.row(i) = ogCenter;
    //     newCenters.row(i) = newCenter;
    // }


    // for (int i = 0; i < mesh_data.F.rows(); i++) {
    //     Eigen::MatrixXd V1(0, 3);
    //     Eigen::MatrixXd V2(0, 3);
    //     int size = faceSize(mesh_data.F.row(i));
    //
    //     // Eigen::Vector3d ogCenter = Eigen::Vector3d::Zero();
    //     // Eigen::Vector3d newCenter = Eigen::Vector3d::Zero();
    //
    //     // std::set<int> hoodFaces;
    //     // for (int j = 0; j < size; j++) {
    //     //     for (auto face: mesh_data.VertPolygons[mesh_data.F(i, j)]) {
    //     //         hoodFaces.insert(face);
    //     //     }
    //     // }
    //     // for (auto face: hoodFaces) {
    //     //     if (face == i) {
    //     //         continue;
    //     //     }
    //     //     V1.conservativeResize(V1.rows() + 1, 3);
    //     //     V2.conservativeResize(V2.rows() + 1, 3);
    //     //     V1.row(V1.rows() - 1) = ogCenters.row(face) - ogCenters.row(i);
    //     //     // V2.row(V2.rows() - 1) = ogCenters.row(face) - ogCenters.row(i);
    //     //     V2.row(V2.rows() - 1) = ogCenters.row(face) - ogCenters.row(i);
    //     // }
    //
    //     for (int j = 0; j < size; j++) {
    //         int corner = mesh_data.F(i, j);
    //         Eigen::Vector3d cornerVert = nV.segment(3 * corner, 3);
    //
    //         // V1.row(V1.rows() - 1) = mesh_data.V.row(corner).transpose() - ogCenter;
    //         // V2.row(V2.rows() - 1) = (mesh_data.V.row(corner).transpose() - ogCenter);
    //         // V2.row(V2.rows() - 1) = cornerVert - newCenter;
    //         // Eigen::Vector3d combined = Eigen::Vector3d::Zero();
    //         // int n = 0;
    //         for (auto k: mesh_data.Hoods[corner]) {
    //             bool sameFace = false;
    //             for (int face: mesh_data.VertPolygons[k]) {
    //                 if (face == i) {
    //                     sameFace = true;
    //                     break;
    //                 }
    //             }
    //             if (sameFace) {
    //                 continue;
    //             }
    //             V1.conservativeResize(V1.rows() + 1, 3);
    //             V2.conservativeResize(V2.rows() + 1, 3);
    //
    //             Eigen::Vector3d vert = nV.segment(3 * k, 3);
    //
    //             //double originalLength = (mesh_data.V.row(corner) - mesh_data.V.row(k)).norm();
    //             // combined += 0.8 * (cornerVert) + 0.2 * (vert + (cornerVert - vert).normalized() * originalLength) -
    //             //         newCenter;
    //
    //             V1.row(V1.rows() - 1) = mesh_data.V.row(k) - ogCenters.row(i);
    //             V2.row(V2.rows() - 1) = vert - newCenters.row(i).transpose();
    //             // n++;
    //         }
    //         // combined = combined / n;
    //         // V2.row(V2.rows() - 1) = combined;
    //
    //         //test
    //
    //         // for (int k: mesh_data.Hoods[corner]) {
    //         //     bool sameFace = false;
    //         //     for (int face: mesh_data.VertPolygons[k]) {
    //         //         if (face == i) {
    //         //             sameFace = true;
    //         //             break;
    //         //         }
    //         //     }
    //         //     if (sameFace) {
    //         //         continue;
    //         //     }
    //         //
    //         //
    //         //     Eigen::Vector3d vert = nV.segment(3 * k, 3);
    //         //
    //         //     double originalLength = (mesh_data.V.row(corner) - mesh_data.V.row(k)).norm();
    //         //
    //         //
    //         //     // add to V1 and V2
    //         //     V1.conservativeResize(V1.rows() + 1, 3);
    //         //     V2.conservativeResize(V2.rows() + 1, 3);
    //         //     V1.row(V1.rows() - 1) = mesh_data.V.row(corner).transpose() - ogCenter;
    //         //     // just placeholder
    //         //     // V2.row(V2.rows() - 1) = mesh_data.V.row(corner).transpose() - ogCenter;
    //         //     V2.row(V2.rows() - 1) = (vert + (cornerVert - vert).normalized() * originalLength) - newCenter;
    //         //     // V2.row(V2.rows() - 1) = (cornerVert + (vert + (cornerVert - vert).normalized() * originalLength)) / 2 -
    //         //     //                         newCenter;
    //         //     // V2.row(V2.rows() - 1) = cornerVert - newCenter;
    //         //     // V2.row(V2.rows() - 1) = mesh_data.V.row(corner).transpose() - ogCenter;
    //         //     // std::cout << ogCenter << std::endl;
    //         //     // std::cout << newCenter << std::endl;
    //         //     // std::cout << mesh_data.V.row(corner) << std::endl;
    //         //     // std::cout << cornerVert << std::endl;
    //         // }
    //     }
    //
    //     //V1.row(j) = ; original
    //     //V2.row(j) = ; current
    //     // }
    //     Eigen::Matrix3d s = V1.transpose() * V2;
    //     for (int x = 0; x < 3; x++) {
    //         for (int y = 0; y < 3; y++) {
    //             S(x * mesh_data.F.rows() + i, y) = s(x, y);
    //         }
    //     }
    // }
    S /= S.array().abs().maxCoeff();


    const int Rdim = 3;
    MatrixXd R(Rdim, 3 * mesh_data.V.rows());
    igl::fit_rotations(S, true, R);


    // 1-degree rotation matrix around Z-axis


    //N = Eigen::MatrixXd::Zero(mesh_data.F.rows(), mesh_data.V.rows() * 3);
    // for (int i = 0; i < mesh_data.F.rows(); i++) {
    //     Matrix3d rot1 = R.block<3, 3>(0, i * 3);
    //     // if (i == 1) {
    //     //     rot1 = rotationMatrix;
    //     // }
    //     // Matrix3d rot1 = rotationMatrix;
    //     Eigen::Vector3d normal = data.Polygons.row(i).head(3);
    //     normal = (rot1 * normal).normalized();
    //     mesh_data.Polygons(i, 0) = normal(0);
    //     mesh_data.Polygons(i, 1) = normal(1);
    //     mesh_data.Polygons(i, 2) = normal(2);
    //     // Eigen::Vector3d normal = mesh_data.Polygons.row(i).head(3);
    //     int size = faceSize(mesh_data.F.row(i));
    //     for (int j = 0; j < size; j++) {
    //         int vert = mesh_data.F(i, j);
    //         N.block<1, 3>(i, 3 * vert) = normal.transpose() / size;
    //     }
    // }


    // NInv = N.completeOrthogonalDecomposition().pseudoInverse();
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
        // Matrix3d vRot = Matrix3d::Zero();
        Eigen::Vector3d rightSide = Eigen::Vector3d::Zero();
        // std::map<int, std::set<int> > vertContr;
        // for (auto p: mesh_data.VertPolygons[i]) {
        //     Eigen::Matrix3d rot = R.block<3, 3>(0, p * 3);
        //     // if (p == 1) {
        //     //     rot = rotationMatrix;
        //     // }
        //     // Eigen::Matrix3d rot = rotationMatrix;
        //     int size = faceSize(mesh_data.F.row(p));
        //     for (int v = 0; v < size; v++) {
        //         if (mesh_data.F(p, (v + 1) % size) == i) {
        //             int before = mesh_data.F(p, v);
        //             int after = mesh_data.F(p, (v + 2) % size);
        //             rightSide -= 0.5 * (rot * (mesh_data.V.row(before) - mesh_data.V.row(i)).transpose());
        //             rightSide -= 0.5 * (rot * (mesh_data.V.row(after) - mesh_data.V.row(i)).transpose());
        //         }
        //     }
        //     //vRot += rot / 3;
        // }

        for (auto v: mesh_data.Hoods[i]) {
            // if (data.positions[v * 3] < 0) {
            //     // rightSide += mesh_data.V.row(i);
            //     rightSide += bc.row(-(data.positions[v * 3] + 1)).transpose();
            // }
            //Eigen::Vector3d vec = (mesh_data.V.row(v) - mesh_data.V.row(i)).transpose();
            rightSide -= rot * (mesh_data.V.row(v) - mesh_data.V.row(i)).transpose();
        }
        for (int j = 0; j < gons.size(); j++) {
            int deletedPoly = gons[j];
            rightSide(0) -= M(3 * i, deletedPoly) * dists[j];
            rightSide(1) -= M(3 * i + 1, deletedPoly) * dists[j];
            rightSide(2) -= M(3 * i + 2, deletedPoly) * dists[j];
        }
        //rightSide(1) = rightSide(1) * 5;
        b(row * 3) = rightSide(0);
        b(row * 3 + 1) = rightSide(1);
        b(row * 3 + 2) = rightSide(2);
        row++;
    }

    //construct Matrix without
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
    // for (int i = 0; i < data.b.size(); i++) {
    //     b(data.b(i) * 3) = bc(i, 0) * 1;
    //     b(data.b(i) * 3 + 1) = bc(i, 1) * 1;
    //     b(data.b(i) * 3 + 2) = bc(i, 2) * 1;
    // }
    // for (int i = 0; i < data.b.size(); i++) {
    //     b(3 * mesh_data.V.rows() + 3 * i) = bc(i, 0) * 100;
    //     b(3 * mesh_data.V.rows() + 3 * i + 1) = bc(i, 1) * 100;
    //     b(3 * mesh_data.V.rows() + 3 * i + 2) = bc(i, 2) * 100;
    // }

    // std::cout << data.L << std::endl;
    // std::cout << b << std::endl;

    // std::cout << "before" << std::endl;
    // Eigen::VectorXd testVerts = NInv * d;
    // std::cout << testVerts << std::endl;
    //TODO: find actual solution
    // for (int i = data.b.size() - 1; i >= 0; i--) {
    //     removeRow(NInv, data.b(i) * 3 + 2);
    //     removeRow(NInv, data.b(i) * 3 + 1);
    //     removeRow(NInv, data.b(i) * 3);
    // }
    // std::cout << "after" << std::endl;
    // testVerts = NInv * d;
    // std::cout << testVerts << std::endl;
    // for (int i = 0; i < data.b.size(); i++) {
    //     Eigen::Vector4d vecs[3];
    //     int j = 0;
    //     for (auto k: mesh_data.VertPolygons[data.b(i)]) {
    //         vecs[j] = mesh_data.Polygons.row(k);
    //         j++;
    //     }
    //     Eigen::Vector3d normal1 = vecs[0].head(3).normalized();
    //     Eigen::Vector3d normal2 = vecs[1].head(3).normalized();
    //     Eigen::Vector3d normal3 = vecs[2].head(3).normalized();
    //
    //     Eigen::Matrix3d m;
    //     m.row(0) = normal1;
    //     m.row(1) = normal2;
    //     m.row(2) = normal3;
    //
    //     Eigen::Vector3d dist = m * bc.row(i).transpose();
    //     j = 0;
    //     for (auto k: mesh_data.VertPolygons[data.b(i)]) {
    //         bestDistances(k) = dist(j);
    //         j++;
    //     }
    // }

    // for (int i = 0; i < bestDistances.size(); i++) {
    //     mesh_data.Polygons(i, 3) = bestDistances(i);
    // }
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

    //test


    // Eigen::VectorXd vertVec(data.V.rows() * 3);
    // for (int i = 0; i < data.V.rows(); i++) {
    //     for (int j = 0; j < 3; j++) {
    //         vertVec(i * 3 + j) = data.V(i, j);
    //     }
    // }


    return true;
}
