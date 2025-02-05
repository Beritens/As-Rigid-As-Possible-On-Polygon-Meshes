//
// Created by ben on 25.09.24.
//

#include "../include/plane_arap.h"


#include <cassert>

#include <iostream>


template<typename Scalar>
using MatrixXX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

void insertInL(Eigen::MatrixXd &L, int v1, int v2, double cot) {
    int a = 3 * v1;
    int b = 3 * v2;
    L(a, b) -= cot;
    L(a + 1, b + 1) -= cot;
    L(a + 2, b + 2) -= cot;

    L(b, a) -= cot;
    L(b + 1, a + 1) -= cot;
    L(b + 2, a + 2) -= cot;

    L(a, a) += cot;
    L(a + 1, a + 1) += cot;
    L(a + 2, a + 2) += cot;

    L(b, b) += cot;
    L(b + 1, b + 1) += cot;
    L(b + 2, b + 2) += cot;
}

double getTriangleArea(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c) {
    return 0.5 * ((b - a).cross(c - a)).norm();
}

bool plane_arap_precomputation(
    poly_mesh_data &mesh_data,
    plane_arap_data &data,
    const Eigen::VectorXi &b) {
    using namespace std;
    using namespace Eigen;
    // number of vertices
    const int n = mesh_data.originalV.rows();


    data.cotanWeights.clear();
    data.lagrangeMultipliers.clear();
    data.extra_grad_planes.clear();
    data.cotanWeights.resize(mesh_data.originalV.rows());

    std::vector<double> areaRatio;

    for (int i = 0; i < mesh_data.F.size(); i++) {
        double coveredArea = 0;
        double fullArea = 0;
        for (int j = 0; j < mesh_data.F[i].size(); j++) {
            int next = (j + 1) % mesh_data.F[i].size();
            int nextnext = (j + 2) % mesh_data.F[i].size();
            Eigen::Vector3d a = mesh_data.originalV.row(mesh_data.F[i][j]);
            Eigen::Vector3d b = mesh_data.originalV.row(mesh_data.F[i][next]);
            Eigen::Vector3d c = mesh_data.originalV.row(mesh_data.F[i][nextnext]);
            double area = getTriangleArea(a, b, c);
            coveredArea += 0.5 * area;
        }
        for (int j = 0; j < mesh_data.triangles[i].size(); j++) {
            Eigen::Vector3d a = mesh_data.originalV.row(mesh_data.triangles[i][j][0]);
            Eigen::Vector3d b = mesh_data.originalV.row(mesh_data.triangles[i][j][1]);
            Eigen::Vector3d c = mesh_data.originalV.row(mesh_data.triangles[i][j][2]);
            double area = getTriangleArea(a, b, c);
            fullArea += area;
        }
        areaRatio.push_back(coveredArea / fullArea);
    }

    for (int i = 0; i < mesh_data.originalV.rows(); i++) {
        for (int j = 0; j < mesh_data.VertNeighbors[i].size(); j++) {
            int next = (j + 1) % mesh_data.VertNeighbors[i].size();
            int faceIdx = -1;
            for (int face: mesh_data.FaceNeighbors[i]) {
                int count = 0;
                for (int l = 0; l < mesh_data.F[face].size(); l++) {
                    if (mesh_data.F[face][l] == i) {
                        count++;
                    }

                    if (mesh_data.F[face][l] == mesh_data.VertNeighbors[i][j]) {
                        count++;
                    }

                    if (mesh_data.F[face][l] == mesh_data.VertNeighbors[i][next]) {
                        count++;
                    }
                }
                if (count >= 3) {
                    faceIdx = face;
                    break;
                }
            }
            Eigen::Vector3d v_a = mesh_data.originalV.row(i);
            Eigen::Vector3d v_b = mesh_data.originalV.row(mesh_data.VertNeighbors[i][j]);
            Eigen::Vector3d v_c = mesh_data.originalV.row(mesh_data.VertNeighbors[i][next]);

            double angleA = getAngle(v_b - v_a, v_c - v_a);
            double angleB = getAngle(v_a - v_b, v_c - v_b);
            double angleC = getAngle(v_a - v_c, v_b - v_c);
            data.cotanWeights[i].push_back((1.0 / tan(angleA)));
            data.cotanWeights[i].push_back((1.0 / tan(angleB)));
            data.cotanWeights[i].push_back((1.0 / tan(angleC)));
        }
    }

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * mesh_data.originalV.rows(),
                                              3 * mesh_data.originalV.rows());
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        std::vector val = mesh_data.VertNeighbors[i];
        int size = val.size();

        for (int j = 0; j < size; j++) {
            int next = (j + 1) % size;
            int n1 = mesh_data.VertNeighbors[i][j];
            int n2 = mesh_data.VertNeighbors[i][next];
            insertInL(L, i, n1, data.cotanWeights[i][j * 3 + 2]);
            insertInL(L, i, n2, data.cotanWeights[i][j * 3 + 1]);
            insertInL(L, n1, n2, data.cotanWeights[i][j * 3]);

            data.edges.push_back({i, i, n1, data.cotanWeights[i][j * 3 + 2]});
            data.edges.push_back({i, i, n2, data.cotanWeights[i][j * 3 + 1]});
            data.edges.push_back({i, n1, n2, data.cotanWeights[i][j * 3]});
        }
    }


    data.L = L.sparseView(1e-9);
    data.Polygons = mesh_data.Planes;
    data.b = b;
    data.V = mesh_data.originalV;
    data.R = Eigen::MatrixXd(3, mesh_data.V.rows() * 3);

    data.conP.clear();
    for (int i = 0; i < data.b.size(); i++) {
        for (auto k: mesh_data.FaceNeighbors[data.b(i)]) {
            data.conP.push_back(k);
        }
    }
    data.distPos.clear();
    int index = 0;
    for (int i = 0; i < mesh_data.F.size(); i++) {
        bool skip = false;
        for (int j = 0; j < data.conP.size(); j++) {
            if (i == data.conP[j]) {
                data.distPos.push_back(-i - 1);

                skip = true;
            }
        }
        if (skip) {
            continue;
        }
        data.distPos.push_back(index);
        index++;
    }

    for (int i = 0; i < data.V.rows(); i++) {
        std::vector<int> polygons = mesh_data.FaceNeighbors[i];
        bool isConstraint = false;
        for (int con = 0; con < b.size(); con++) {
            if (b(con) == i) {
                isConstraint = true;
            }
        }
        if (isConstraint) {
            continue;
        }

        // nIdx.push_back(nis);
        for (int c = 3; c < polygons.size(); c++) {
            data.lagrangeMultipliers.push_back(0.0);
            data.extra_grad_planes.push_back({i, c});
        }
    }

    return true;
}

void getRotations(poly_mesh_data &mesh_data, plane_arap_data &data) {
    for (int i = 0; i < data.V.rows(); i++) {
        Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(mesh_data.VertNeighbors[i].size() * 3, 3);
        Eigen::MatrixXd V2 = Eigen::MatrixXd::Zero(mesh_data.VertNeighbors[i].size() * 3, 3);

        Eigen::Vector3d originalVert = data.V.row(i);
        Eigen::Vector3d newVert = mesh_data.V.row(i);

        int size = mesh_data.VertNeighbors[i].size();

        for (int j = 0; j < size; j++) {
            int next = (j + 1) % size;
            int n1 = mesh_data.VertNeighbors[i][j];
            int n2 = mesh_data.VertNeighbors[i][next];
            Eigen::Vector3d originalNeighbor = data.V.row(n1);
            Eigen::Vector3d newNeighbor = mesh_data.V.row(n1);
            Eigen::Vector3d originalNeighbor2 = data.V.row(n2);
            Eigen::Vector3d newNeighbor2 = mesh_data.V.row(n2);
            V1.row(j * 3) = data.cotanWeights[i][j * 3 + 2] * (originalVert - originalNeighbor);
            V2.row(j * 3) = data.cotanWeights[i][j * 3 + 2] * (newVert - newNeighbor);

            V1.row(j * 3 + 1) = data.cotanWeights[i][j * 3 + 1] * (originalVert - originalNeighbor2);
            V2.row(j * 3 + 1) = data.cotanWeights[i][j * 3 + 1] * (newVert - newNeighbor2);

            V1.row(j * 3 + 2) = data.cotanWeights[i][j * 3] * (originalNeighbor - originalNeighbor2);
            V2.row(j * 3 + 2) = data.cotanWeights[i][j * 3] * (newNeighbor - newNeighbor2);
        }


        Eigen::Matrix3d rot = getRotation(V1, V2);
        data.R.block<3, 3>(0, i * 3) = rot;
    }
}

void insertInB(Eigen::VectorXd &b, int v1, int v2, double cot, const Eigen::MatrixXd V, const Eigen::Matrix3d rot) {
    Eigen::Vector3d rightSide = cot * (rot * (V.row(v2) - V.row(v1)).transpose());

    int p1 = v1;
    int p2 = v2;
    b(p1 * 3) -= rightSide(0);
    b(p1 * 3 + 1) -= rightSide(1);
    b(p1 * 3 + 2) -= rightSide(2);

    b(p2 * 3) += rightSide(0);
    b(p2 * 3 + 1) += rightSide(1);
    b(p2 * 3 + 2) += rightSide(2);
}


bool global_distance_step(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    using namespace Eigen;
    using namespace std;

    const int n = data.V.rows();
    // std::vector<Eigen::MatrixXd> invNs;
    // std::vector<std::vector<int> > nIdx;

    std::vector<Eigen::MatrixXd> invNs;
    std::vector<std::vector<int> > nIdx;


    for (int i = 0; i < data.V.rows(); i++) {
        std::vector<int> polygons = mesh_data.FaceNeighbors[i];

        std::vector<Eigen::MatrixXd> ns;
        std::vector<std::vector<int> > nis;
        MatrixXd NCon(3, 3);
        std::vector<int> idxCon;
        for (int j = 0; j < 3; j++) {
            int pol = polygons[j];
            NCon.row(j) = mesh_data.Planes.row(pol).head(3).normalized();
            idxCon.push_back(pol);
        }
        Eigen::MatrixXd NInvCon = NCon.inverse();

        invNs.push_back(NInvCon);
        nIdx.push_back(idxCon);
    }

    int rows = data.V.rows() * 3;
    int cols = mesh_data.Planes.rows();
    Eigen::SparseMatrix<double> NInv(rows, cols);
    std::vector<Eigen::Triplet<double> > triplets;

    //go over all vertices
    for (int i = 0; i < nIdx.size(); i++) {
        //go over 3 dimensions
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < nIdx[i].size(); k++) {
                triplets.emplace_back(i * 3 + j, nIdx[i][k], invNs[i](j, k));
                // NInv(i * 3 + j, nIdx[i][k]) += invNs[i](j, k);
            }
        }
    }
    NInv.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SparseMatrix<double> NInvT = NInv.transpose();

    Eigen::SparseMatrix<double> M = NInvT * data.L * NInv;

    std::vector<double> dists;

    for (int i = 0; i < data.b.size(); i++) {
        std::vector<Eigen::Vector4d> vecs;
        int j = 0;
        for (auto k: mesh_data.FaceNeighbors[data.b(i)]) {
            vecs.push_back(mesh_data.Planes.row(k));
            j++;
        }
        //
        // Eigen::Vector3d normal1 = vecs[0].head(3).normalized();
        // Eigen::Vector3d normal2 = vecs[1].head(3).normalized();
        // Eigen::Vector3d normal3 = vecs[2].head(3).normalized();
        //
        // Eigen::Matrix3d m;
        // m.row(0) = normal1;
        // m.row(1) = normal2;
        // m.row(2) = normal3;
        //
        // Eigen::Vector3d dist = m * bc.row(i).transpose();
        j = 0;
        for (auto k: mesh_data.FaceNeighbors[data.b(i)]) {
            double dist = vecs[j].head(3).normalized().dot(bc.row(i));
            dists.push_back(dist);
            mesh_data.Planes(k, 3) = dist;
            j++;
        }
    }


    Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * data.V.rows());

    for (int i = 0; i < mesh_data.V.rows(); i++) {
        std::vector val = mesh_data.VertNeighbors[i];
        int size = val.size();

        Eigen::Matrix3d rot = data.R.block<3, 3>(0, i * 3);

        for (int j = 0; j < size; j++) {
            int next = (j + 1) % size;
            int n1 = mesh_data.VertNeighbors[i][j];
            int n2 = mesh_data.VertNeighbors[i][next];
            //anders probieren. einfach laplacian benutzen und dann direkt edge rein addieren I guess
            insertInB(b, i, n1, data.cotanWeights[i][j * 3 + 2], data.V, rot);
            insertInB(b, i, n2, data.cotanWeights[i][j * 3 + 1], data.V, rot);
            insertInB(b, n2, n1, data.cotanWeights[i][j * 3], data.V, rot);
        }
    }

    b = NInvT * b;


    std::vector<Eigen::Triplet<double> > lagrangeTriplets;

    int extra_rows = 0;

    // higher vertex degree stuff
    for (int i = 0; i < data.extra_grad_planes.size(); i++) {
        int vertex_index = data.extra_grad_planes[i].vertex;
        int plane_vertex_index = data.extra_grad_planes[i].plane;
        int plane_index = mesh_data.FaceNeighbors[vertex_index][plane_vertex_index];
        Eigen::MatrixXd invNorm = invNs[vertex_index];
        std::vector<int> distance_indices = nIdx[vertex_index];
        Eigen::Vector3d plane_normal = data.Polygons.row(plane_index).head(3).normalized();

        Eigen::Vector3d coefficient = plane_normal.transpose() * invNorm;

        lagrangeTriplets.emplace_back(extra_rows, plane_index, -1.0);
        lagrangeTriplets.emplace_back(extra_rows, distance_indices[0], coefficient(0));
        lagrangeTriplets.emplace_back(extra_rows, distance_indices[1], coefficient(1));
        lagrangeTriplets.emplace_back(extra_rows, distance_indices[2], coefficient(2));
        extra_rows++;
    }

    // //go over all vertices
    // for (int i = 0; i < nIdx.size(); i++) {
    //     //go over all constraints
    //     for (int c = 1; c < nIdx[i].size(); c++) {
    //         //go over dimensions
    //         // std::cout << invNs[i][0] << std::endl;
    //         // std::cout << invNs[i][c] << std::endl;
    //         for (int d = 0; d < 3; d++) {
    //             //go over data
    //             std::map<int, double> values;
    //             for (int nor = 0; nor < 3; nor++) {
    //                 // mTriplets.emplace_back(newRows + extra_rows + d, nIdx[i][0][nor], invNs[i][0](d, nor));
    //                 values[nIdx[i][0][nor]] += invNs[i][0](d, nor);
    //                 values[nIdx[i][c][nor]] -= invNs[i][c](d, nor);
    //             }
    //
    //             for (const auto &[key, value]: values) {
    //                 lagrangeTriplets.emplace_back(extra_rows, key, value);
    //
    //                 // lagrangeTriplets.emplace_back(key, newRows + extra_rows, value);
    //             }
    //             extra_rows++;
    //         }
    //     }
    // }

    Eigen::SparseMatrix<double> lagrangeM(extra_rows, M.rows());
    lagrangeM.setFromTriplets(lagrangeTriplets.begin(), lagrangeTriplets.end());

    Eigen::SparseMatrix<double> lagrangeT = lagrangeM.transpose();

    Eigen::MatrixXd combinedMatrix = Eigen::MatrixXd::Zero(M.rows() + lagrangeM.rows(), M.cols() + lagrangeM.rows());

    combinedMatrix.topLeftCorner(M.rows(), M.cols()) = M;
    combinedMatrix.topRightCorner(M.rows(), lagrangeT.cols()) = lagrangeT;
    combinedMatrix.bottomLeftCorner(lagrangeM.rows(), M.cols()) = lagrangeM;

    //triplets later used for the final matrix
    std::vector<Eigen::Triplet<double> > mTriplets;


    int newRows = combinedMatrix.rows() - data.conP.size();
    int newCols = combinedMatrix.cols() - data.conP.size();


    for (int i = 0; i < combinedMatrix.rows(); ++i) {
        for (int j = 0; j < combinedMatrix.cols(); ++j) {
            double value = combinedMatrix(i, j);
            int row = i >= data.distPos.size()
                          ? i - data.conP.size()
                          : data.distPos[i];
            int col = j >= data.distPos.size()
                          ? j - data.conP.size()
                          : data.distPos[j];
            if (row >= 0 && col >= 0) {
                mTriplets.emplace_back(row, col, value);
            }
        }
    }


    Eigen::VectorXd newB(b.size() - data.conP.size() + extra_rows);
    //delete rows and cols and update b to add constraints
    for (int i = 0; i < b.size(); i++) {
        if (data.distPos[i] < 0) {
            continue;
        }
        newB(data.distPos[i]) = b(i);
        for (int j = 0; j < data.conP.size(); j++) {
            int deletedIndex = data.conP[j];
            newB(data.distPos[i]) -= combinedMatrix.coeff(i, deletedIndex) * dists[j];
        }
    }
    for (int i = 0; i < extra_rows; i++) {
        int old_index = b.size() + i;
        int new_index = b.size() - data.conP.size() + i;
        newB(new_index) = 0.0;
        for (int j = 0; j < data.conP.size(); j++) {
            int deletedIndex = data.conP[j];
            newB(new_index) -= combinedMatrix.coeff(old_index, deletedIndex) * dists[j];
        }
    }


    Eigen::SparseMatrix<double> newM(newRows, newCols);
    newM.setFromTriplets(mTriplets.begin(), mTriplets.end());

    //constraints cover whole mesh
    if (newM.size() == 0) {
        return true;
    }

    // std::cout << newM << std::endl;
    // std::cout << newB << std::endl;

    // Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
    // solver.compute(newM);
    // Eigen::VectorXd bestDistances = solver.solve(newB);
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver;
    // solver.compute(newM);
    // solver.setMaxIterations(100000);
    // Eigen::VectorXd bestDistances = solver.solve(newB);

    Eigen::SparseMatrix<double> identity(newM.rows(), newM.cols());
    identity.setIdentity();
    newM += identity * 0.00000001;

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
    solver.compute(newM);
    if (solver.info() != Eigen::Success) {
        std::cout << "decomposition failed" << std::endl;
        return false;
    }
    Eigen::VectorXd bestDistances = solver.solve(newB);
    // std::cout << bestDistances << std::endl;

    // std::cout << newM << std::endl;
    for (int i = 0; i < mesh_data.Planes.rows(); i++) {
        bool skipped = data.distPos[i] < 0;
        if (skipped) {
            continue;
        }

        mesh_data.Planes(i, 3) = bestDistances(data.distPos[i]);
    }

    for (int i = 0; i < extra_rows; i++) {
        data.lagrangeMultipliers[i] = bestDistances[b.size() - data.conP.size() + i];
    }

    return true;
}

TinyAD::ScalarFunction<4, double, long long> getFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    TinyAD::ScalarFunction<4, double, long long> func = TinyAD::scalar_function<4>(TinyAD::range(data.Polygons.rows()));

    func.add_elements<10>(TinyAD::range(mesh_data.V.rows()),
                          [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                              //calculate arap energy
                              using T = TINYAD_SCALAR_TYPE(


                                  element
                              );

                              //projection calculation

                              Eigen::Index v_idx = element.handle;
                              // std::vector<int> localConstrainsIndex;
                              std::map<int, int> constrained_face;
                              for (int i = 0; i < data.b.size(); i++) {
                                  for (auto v: mesh_data.VertNeighbors[v_idx]) {
                                      for (auto p: mesh_data.FaceNeighbors[v]) {
                                          int size = mesh_data.F[p].size();
                                          for (int fi = 0; fi < size; fi++) {
                                              if (mesh_data.F[p][fi] == data.b(i)) {
                                                  constrained_face[p] = i;
                                              }
                                          }
                                      }
                                  }
                              }

                              std::map<int, Eigen::Vector4<T> > gons;

                              for (auto const &[key, val]: constrained_face) {
                                  Eigen::Vector3<T> normal = element.variables(key).head(3).normalized();
                                  Eigen::Vector3d point = bc.row(val);
                                  T dist = normal.dot(point);
                                  gons[key] = Eigen::Vector4<T>(normal(0), normal(1), normal(2), dist);
                              }
                              // for (int i = 0; i < localConstrainsIndex.size(); i++) {
                              //     Eigen::Vector4<T> vecs[3];
                              //     int j = 0;
                              //     for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                              //         vecs[j] = element.variables(k);
                              //         j++;
                              //     }
                              //     Eigen::Vector3<T> normal1 = vecs[0].head(3).normalized();
                              //     Eigen::Vector3<T> normal2 = vecs[1].head(3).normalized();
                              //     Eigen::Vector3<T> normal3 = vecs[2].head(3).normalized();
                              //
                              //     Eigen::Matrix3<T> m;
                              //     m.row(0) = normal1;
                              //     m.row(1) = normal2;
                              //     m.row(2) = normal3;
                              //
                              //     Eigen::Vector3<T> dist =
                              //             m * bc.row(localConstrainsIndex[i]).transpose();
                              //     j = 0;
                              //     for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                              //         Eigen::Vector4<T> pol = element.variables(k);
                              //         pol(3) = dist(j);
                              //         gons[k] = pol;
                              //         j++;
                              //     }
                              // }

                              std::vector<Eigen::Vector4<T> > polygons;
                              for (auto f: mesh_data.FaceNeighbors[v_idx]) {
                                  Eigen::Vector4<T> pol;
                                  if (gons.find(f) != gons.end()) {
                                      pol = gons[f];
                                  } else {
                                      pol = element.variables(f);
                                  }
                                  polygons.push_back(pol);
                              }
                              Eigen::Vector3<T> vert = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                              Eigen::Vector3d ogVert = data.V.row(v_idx);
                              std::vector<Eigen::Vector3<T> > points;


                              //simply go over neighbours and sum over triangles

                              for (auto neighbor: mesh_data.VertNeighbors[v_idx]) {
                                  std::vector<Eigen::Vector4<T> > neighPolygons;
                                  for (auto f: mesh_data.FaceNeighbors[neighbor]) {
                                      Eigen::Vector4<T> pol;
                                      if (gons.find(f) != gons.end()) {
                                          pol = gons[f];
                                      } else {
                                          pol = element.variables(f);
                                      }
                                      neighPolygons.push_back(pol);
                                  }
                                  Eigen::Vector3<T> neighborVert = getPoint<T>(
                                      neighPolygons[0], neighPolygons[1], neighPolygons[2]);
                                  points.push_back(neighborVert);
                              }

                              int size = mesh_data.VertNeighbors[v_idx].size();
                              Eigen::Matrix3d Rot = data.R.block<3, 3>(0, v_idx * 3);

                              T returnValue = 0;
                              for (int j = 0; j < size; j++) {
                                  int next = (j + 1) % size;
                                  int n1 = mesh_data.VertNeighbors[v_idx][j];
                                  int n2 = mesh_data.VertNeighbors[v_idx][next];
                                  Eigen::Vector3d ogNeighbor = data.V.row(n1);
                                  Eigen::Vector3d ogNeighbor2 = data.V.row(n2);
                                  Eigen::Vector3d v = ogNeighbor - ogVert;
                                  Eigen::Vector3<T> tv = points[j] - vert;

                                  Eigen::Vector3d v2 = ogNeighbor2 - ogVert;
                                  Eigen::Vector3<T> tv2 = points[next] - vert;

                                  Eigen::Vector3d v3 = ogNeighbor2 - ogNeighbor;
                                  Eigen::Vector3<T> tv3 = points[next] - points[j];

                                  returnValue += data.cotanWeights[v_idx][j * 3 + 2] * (tv - Rot * v).squaredNorm();
                                  returnValue += data.cotanWeights[v_idx][j * 3 + 1] * (tv2 - Rot * v2).squaredNorm();
                                  returnValue += data.cotanWeights[v_idx][j * 3] * (tv3 - Rot * v3).squaredNorm();
                              }
                              return returnValue;
                          });
    return func;
}

TinyAD::ScalarFunction<3, double, long long> getBlockFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    TinyAD::ScalarFunction<3, double, long long> func = TinyAD::scalar_function<3>(TinyAD::range(data.Polygons.rows()));

    func.add_elements<10>(TinyAD::range(mesh_data.V.rows()),
                          [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                              //calculate arap energy
                              using T = TINYAD_SCALAR_TYPE(


                                  element
                              );

                              Eigen::Index v_idx = element.handle;
                              std::vector<int> localConstrainsIndex;
                              for (int i = 0; i < data.b.size(); i++) {
                                  if (v_idx == data.b(i)) {
                                      localConstrainsIndex.push_back(i);
                                  }
                                  for (auto v: mesh_data.VertNeighbors[v_idx]) {
                                      for (auto p: mesh_data.FaceNeighbors[v]) {
                                          int size = mesh_data.F[p].size();
                                          for (int fi = 0; fi < size; fi++) {
                                              if (mesh_data.F[p][fi] == data.b(i)) {
                                                  localConstrainsIndex.push_back(i);
                                              }
                                          }
                                      }
                                  }
                              }

                              std::map<int, Eigen::Vector4<T> > gons;
                              for (int i = 0; i < localConstrainsIndex.size(); i++) {
                                  Eigen::Vector3<T> vecs[3];
                                  int j = 0;
                                  for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                                      vecs[j] = element.variables(k);
                                      j++;
                                  }
                                  Eigen::Vector3<T> normal1 = vecs[0];
                                  Eigen::Vector3<T> normal2 = vecs[1];
                                  Eigen::Vector3<T> normal3 = vecs[2];

                                  Eigen::Matrix3<T> m;
                                  m.row(0) = normal1;
                                  m.row(1) = normal2;
                                  m.row(2) = normal3;

                                  Eigen::Vector3<T> dist =
                                          m * bc.row(localConstrainsIndex[i]).transpose();
                                  j = 0;
                                  for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                                      Eigen::Vector4<T> pol;
                                      pol.head(3) = element.variables(k);
                                      pol(3) = dist(j);
                                      gons[k] = pol;
                                      j++;
                                  }
                              }

                              std::vector<Eigen::Vector4<T> > polygons;
                              for (auto f: mesh_data.FaceNeighbors[v_idx]) {
                                  Eigen::Vector4<T> pol;
                                  if (gons.find(f) != gons.end()) {
                                      pol = gons[f];
                                  } else {
                                      Eigen::Vector3<T> normal = element.variables(f);
                                      pol.head(3) = normal;
                                      pol(3) = mesh_data.Planes(f, 3);
                                  }
                                  polygons.push_back(pol);
                              }
                              Eigen::Vector3<T> vert = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                              Eigen::Vector3d ogVert = data.V.row(v_idx);
                              std::vector<Eigen::Vector3<T> > points;

                              for (auto neighbor: mesh_data.VertNeighbors[v_idx]) {
                                  std::vector<Eigen::Vector4<T> > neighPolygons;
                                  for (auto f: mesh_data.FaceNeighbors[neighbor]) {
                                      Eigen::Vector4<T> pol;
                                      if (gons.find(f) != gons.end()) {
                                          pol = gons[f];
                                      } else {
                                          Eigen::Vector3<T> normal = element.variables(f);
                                          pol.head(3) = normal;
                                          pol(3) = mesh_data.Planes(f, 3);
                                      }
                                      neighPolygons.push_back(pol);
                                  }
                                  Eigen::Vector3<T> neighborVert = getPoint<T>(
                                      neighPolygons[0], neighPolygons[1], neighPolygons[2]);
                                  points.push_back(neighborVert);
                              }

                              int size = mesh_data.VertNeighbors[v_idx].size();

                              Eigen::Matrix3d Rot = data.R.block<3, 3>(0, v_idx * 3);

                              T returnValue = 0;
                              for (int j = 0; j < size; j++) {
                                  int next = (j + 1) % size;
                                  int n1 = mesh_data.VertNeighbors[v_idx][j];
                                  int n2 = mesh_data.VertNeighbors[v_idx][next];
                                  Eigen::Vector3d ogNeighbor = data.V.row(n1);
                                  Eigen::Vector3d ogNeighbor2 = data.V.row(n2);
                                  Eigen::Vector3d v = ogNeighbor - ogVert;
                                  Eigen::Vector3<T> tv = points[j] - vert;

                                  Eigen::Vector3d v2 = ogNeighbor2 - ogVert;
                                  Eigen::Vector3<T> tv2 = points[next] - vert;

                                  Eigen::Vector3d v3 = ogNeighbor2 - ogNeighbor;
                                  Eigen::Vector3<T> tv3 = points[next] - points[j];

                                  returnValue += data.cotanWeights[v_idx][j * 3 + 2] * (tv - Rot * v).squaredNorm();
                                  returnValue += data.cotanWeights[v_idx][j * 3 + 1] * (tv2 - Rot * v2).squaredNorm();
                                  returnValue += data.cotanWeights[v_idx][j * 3] * (tv3 - Rot * v3).squaredNorm();
                              }
                              return returnValue;
                          });
    return func;
}

TinyAD::ScalarFunction<4, double, long long> getConstraintFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data,
    int index) {
    TinyAD::ScalarFunction<4, double, long long> func = TinyAD::scalar_function<4>(
        TinyAD::range(data.Polygons.rows()));
    func.add_elements<5>(TinyAD::range(1),
                         [&,index](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                             using T = TINYAD_SCALAR_TYPE(
                                 element
                             );

                             int vertex_idx = data.extra_grad_planes[index].vertex;
                             int plane_vertex_idx = data.extra_grad_planes[index].plane;
                             int plane_idx = mesh_data.FaceNeighbors[vertex_idx][plane_vertex_idx];

                             std::vector<int> face_indices = mesh_data.FaceNeighbors[vertex_idx];

                             Eigen::Vector3<T> point = getPoint<T>(element.variables(face_indices[0]),
                                                                   element.variables(face_indices[1]),
                                                                   element.variables(face_indices[2]));
                             Eigen::Vector4<T> extra_plane = element.variables(plane_idx);
                             Eigen::Vector3<T> normal = extra_plane.head(3).normalized();
                             T dist = normal.dot(point);
                             return (dist - extra_plane(3));
                             // return 1000000.0 * (dist - extra_plane(3)) * (dist - extra_plane(3));
                         });
    return func;
}

TinyAD::ScalarFunction<4, double, long long> getEdgeFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    TinyAD::ScalarFunction<4, double, long long> func = TinyAD::scalar_function<4>(
        TinyAD::range(data.Polygons.rows()));

    // func.add_elements<5>(TinyAD::range(data.extra_grad_planes.size()),
    //                      [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    //                          using T = TINYAD_SCALAR_TYPE(
    //                              element
    //                          );
    //
    //                          int e_idx = element.handle;
    //                          int vertex_idx = data.extra_grad_planes[e_idx].vertex;
    //                          int plane_vertex_idx = data.extra_grad_planes[e_idx].plane;
    //                          int plane_idx = mesh_data.FaceNeighbors[vertex_idx][plane_vertex_idx];
    //
    //                          std::vector<int> face_indices = mesh_data.FaceNeighbors[vertex_idx];
    //
    //                          Eigen::Vector3<T> point = getPoint<T>(element.variables(face_indices[0]),
    //                                                                element.variables(face_indices[1]),
    //                                                                element.variables(face_indices[2]));
    //                          Eigen::Vector4<T> extra_plane = element.variables(plane_idx);
    //                          Eigen::Vector3<T> normal = extra_plane.head(3).normalized();
    //                          T dist = normal.dot(point);
    //                          return element.variables(data.Polygons.rows() + e_idx)(0) * (dist - extra_plane(3));
    //                          // return 1000000.0 * (dist - extra_plane(3)) * (dist - extra_plane(3));
    //                      });

    func.add_elements<6>(TinyAD::range(data.edges.size()),
                         [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                             //not used. tried to sum over edges instead of rotation cells but this was a lot slower
                             using T = TINYAD_SCALAR_TYPE(
                                 element
                             );

                             Eigen::Index e_idx = element.handle;
                             edge e = data.edges[e_idx];
                             std::vector<Eigen::Vector3d> localConstrains;
                             std::vector<int> localConstrainsIndex;
                             std::vector<Eigen::Vector3<T> > localConstraintsNormals;


                             std::vector<int> neighborsA = mesh_data.FaceNeighbors[e.a];
                             std::vector<int> neighborsB = mesh_data.FaceNeighbors[e.b];

                             std::vector<int> faces;
                             faces.reserve(6); // Reserve space for efficiency

                             for (size_t i = 0; i < std::min<size_t>(3, neighborsA.size()); ++i) {
                                 faces.push_back(neighborsA[i]);
                             }

                             for (size_t i = 0; i < std::min<size_t>(3, neighborsB.size()); ++i) {
                                 faces.push_back(neighborsB[i]);
                             }

                             std::sort(faces.begin(), faces.end());
                             faces.erase(std::unique(faces.begin(), faces.end()), faces.end());

                             for (int i = 0; i < data.b.size(); i++) {
                                 // if (e.a == data.b(i) || e.b == data.b(i)) {
                                 //     localConstrains.push_back(bc.row(i));
                                 // }
                                 for (auto p: faces) {
                                     int size = mesh_data.F[p].size();
                                     for (int fi = 0; fi < size; fi++) {
                                         if (mesh_data.F[p][fi] == data.b(i)) {
                                             localConstrains.push_back(bc.row(i));
                                             localConstrainsIndex.push_back(p);
                                             localConstraintsNormals.push_back(
                                                 element.variables(p).head(3).normalized());
                                         }
                                     }
                                 }
                             }

                             std::map<int, Eigen::Vector4<T> > gons;
                             for (int i = 0; i < localConstrains.size(); i++) {
                                 Eigen::Vector3<T> normal = localConstraintsNormals[i];
                                 Eigen::Vector3d point = localConstrains[i];
                                 T distance = normal.dot(point);
                                 int index = localConstrainsIndex[i];

                                 Eigen::Vector4<T> pol(normal(0), normal(1), normal(2), distance);
                                 gons[index] = pol;
                             }
                             // for (int i = 0; i < localConstrainsIndex.size(); i++) {
                             //     Eigen::Vector4<T> vecs[3];
                             //     int j = 0;
                             //     for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                             //         vecs[j] = element.variables(k);
                             //         j++;
                             //     }
                             //     Eigen::Vector3<T> normal1 = vecs[0].head(3).normalized();
                             //     Eigen::Vector3<T> normal2 = vecs[1].head(3).normalized();
                             //     Eigen::Vector3<T> normal3 = vecs[2].head(3).normalized();
                             //
                             //     Eigen::Matrix3<T> m;
                             //     m.row(0) = normal1;
                             //     m.row(1) = normal2;
                             //     m.row(2) = normal3;
                             //
                             //     Eigen::Vector3<T> dist =
                             //             m * bc.row(localConstrainsIndex[i]).transpose();
                             //     j = 0;
                             //     for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                             //         Eigen::Vector4<T> pol = element.variables(k);
                             //         pol(3) = dist(j);
                             //         gons[k] = pol;
                             //         j++;
                             //     }
                             // }

                             std::vector<Eigen::Vector4<T> > polygons;
                             int j = 0;
                             for (auto f: mesh_data.FaceNeighbors[e.a]) {
                                 Eigen::Vector4<T> pol;
                                 if (gons.find(f) != gons.end()) {
                                     pol = gons[f];
                                 } else {
                                     pol = element.variables(f);
                                 }
                                 polygons.push_back(pol);
                                 j++;
                                 if (j >= 3) {
                                     break;
                                 }
                             }
                             Eigen::Vector3<T> a = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                             Eigen::Vector3d ogA = data.V.row(e.a);

                             polygons.clear();
                             j = 0;
                             for (auto f: mesh_data.FaceNeighbors[e.b]) {
                                 Eigen::Vector4<T> pol;
                                 if (gons.find(f) != gons.end()) {
                                     pol = gons[f];
                                 } else {
                                     pol = element.variables(f);
                                 }
                                 polygons.push_back(pol);
                                 j++;
                                 if (j >= 3) {
                                     break;
                                 }
                             }
                             Eigen::Vector3<T> b = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                             Eigen::Vector3d ogB = data.V.row(e.b);

                             Eigen::Matrix3d Rot = data.R.block<3, 3>(0, e.rot * 3);

                             T returnValue = 0;
                             Eigen::Vector3d v = ogB - ogA;
                             Eigen::Vector3<T> tv = b - a;

                             returnValue += e.w * (tv - Rot * v).squaredNorm();
                             return returnValue;
                         });
    return func;
}
