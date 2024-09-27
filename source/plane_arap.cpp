//
// Created by ben on 25.09.24.
//

#include "../include/plane_arap.h"


#include <cassert>

#include <iostream>


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
    //calculate L matrix
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * mesh_data.V.rows(), 3 * mesh_data.V.rows());
    for (auto const &[key, val]: mesh_data.Hoods) {
        int size = val.size();
        L(key * 3, key * 3) = size;
        L(key * 3 + 1, key * 3 + 1) = size;
        L(key * 3 + 2, key * 3 + 2) = size;
        for (auto j: val) {
            L(key * 3, j * 3) = -1;
            L(key * 3 + 1, j * 3 + 1) = -1;
            L(key * 3 + 2, j * 3 + 2) = -1;
        }
    }
    data.L = L;
    data.Polygons = mesh_data.Polygons;
    return true;
}


bool plane_arap_solve(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    using namespace Eigen;
    using namespace std;
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


    MatrixXd S(mesh_data.F.rows() * 3, 3);
    //rotations:
    for (int i = 0; i < mesh_data.F.rows(); i++) {
        Eigen::MatrixXd V1(0, 3);
        Eigen::MatrixXd V2(0, 3);
        int j = 0;
        int size = faceSize(mesh_data.F.row(i));

        Eigen::Vector3d ogCenter = Eigen::Vector3d::Zero();
        Eigen::Vector3d newCenter = Eigen::Vector3d::Zero();

        for (int j = 0; j < size; j++) {
            ogCenter += mesh_data.V.row(mesh_data.F(i, j)) / size;
            Eigen::Vector3d newVert = nV.segment(mesh_data.F(i, j) * 3, 3);
            newCenter += newVert / size;
        }


        for (int j = 0; j < size; j++) {
            //V1.row(j) = it->second + rot1.inverse()*(-rot2*custom_data.sideVecs[it->first][i]);
            // Eigen::Vector3d normal = rot2 * Polygons.row(it->first).head(3).transpose();
            // Eigen::Vector3d b = custom_data.sideVecs[it->first][i];
            // double length = b.norm();
            // b = rot1 * b;
            // Eigen::Vector3d projected = normal * (normal.dot(b));
            // b = b - projected;
            // b = b.normalized()*length;
            for (int k: mesh_data.Hoods[mesh_data.F(i, j)]) {
                bool sameFace = false;
                for (int face: mesh_data.VertPolygons[k]) {
                    if (face == i) {
                        sameFace = true;
                        break;
                    }
                }
                if (sameFace) {
                    break;
                }

                // add to V1 and V2
                V1.conservativeResize(V1.rows() + 1, 3);
                V2.conservativeResize(V2.rows() + 1, 3);
                V1.row(V1.rows() - 1) = mesh_data.V.row(k) = ogCenter;
                //TODO: add row to V2
            }

            //V1.row(j) = ; original
            //V2.row(j) = ; current
        }
        Eigen::Matrix3d s = V1.transpose() * V2;
        for (int x = 0; x < mesh_data.F.rows(); x++) {
            for (int y = 0; y < 3; y++) {
                S(x * mesh_data.F.rows() + i, y) = s(x, y);
            }
        }
    }
    S /= S.array().abs().maxCoeff();


    const int Rdim = 3;
    MatrixXd R(Rdim, 3 * mesh_data.F.rows());
    igl::fit_rotations(S, true, R);


    N = Eigen::MatrixXd::Zero(mesh_data.F.rows(), mesh_data.V.rows() * 3);
    for (int i = 0; i < mesh_data.F.rows(); i++) {
        Matrix3d rot1 = R.block<3, 3>(0, i * 3);
        Eigen::Vector3d normal = rot1 * data.Polygons.row(i).head(3);
        int size = faceSize(mesh_data.F.row(i));
        for (int j = 0; j < size; j++) {
            int vert = mesh_data.F(i, j);
            N.block<1, 3>(i, 3 * vert) = normal.transpose() / size;
        }
    }
    //Eigen::MatrixXd M = data.L * NInv;


    // Eigen::VectorXd vertVec(data.V.rows() * 3);
    // for (int i = 0; i < data.V.rows(); i++) {
    //     for (int j = 0; j < 3; j++) {
    //         vertVec(i * 3 + j) = data.V(i, j);
    //     }
    // }


    return true;
}
