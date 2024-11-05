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

bool plane_arap_precomputation(
    poly_mesh_data &mesh_data,
    plane_arap_data &data,
    const Eigen::VectorXi &b) {
    using namespace std;
    using namespace Eigen;
    // number of vertices
    const int n = mesh_data.originalV.rows();


    data.cotanWeights.clear();
    data.cotanWeights.resize(mesh_data.originalV.rows());

    for (int i = 0; i < mesh_data.originalV.rows(); i++) {
        for (int j = 0; j < mesh_data.VertNeighbors[i].size(); j++) {
            int next = (j + 1) % mesh_data.VertNeighbors[i].size();
            Eigen::Vector3d v_a = mesh_data.originalV.row(i);
            Eigen::Vector3d v_b = mesh_data.originalV.row(mesh_data.VertNeighbors[i][j]);
            Eigen::Vector3d v_c = mesh_data.originalV.row(mesh_data.VertNeighbors[i][next]);

            double angleA = getAngle(v_b - v_a, v_c - v_a);
            double angleB = getAngle(v_a - v_b, v_c - v_b);
            double angleC = getAngle(v_a - v_c, v_b - v_c);
            // data.cotanWeights[i].push_back(1.0);
            // data.cotanWeights[i].push_back(1.0);
            // data.cotanWeights[i].push_back(1.0);

            data.cotanWeights[i].push_back(1.0 / tan(angleA));
            data.cotanWeights[i].push_back(1.0 / tan(angleB));
            data.cotanWeights[i].push_back(1.0 / tan(angleC));
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
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const int n = data.V.rows();
    std::vector<Eigen::MatrixXd> invNs;
    std::vector<std::vector<int> > nIdx;


    for (int i = 0; i < data.V.rows(); i++) {
        std::set<int> polygons = mesh_data.FaceNeighbors[i];
        MatrixXd N(polygons.size(), 3);
        std::vector<int> idx;
        int j = 0;
        for (auto pol: polygons) {
            N.row(j) = mesh_data.Planes.row(pol).head(3);
            idx.push_back(pol);

            j++;
        }
        Eigen::MatrixXd NInv = N.completeOrthogonalDecomposition().pseudoInverse();

        invNs.push_back(NInv);
        nIdx.push_back(idx);
    }

    int rows = data.V.rows() * 3;
    int cols = mesh_data.Planes.rows();
    Eigen::SparseMatrix<double> NInv(rows, cols);
    std::vector<Eigen::Triplet<double> > triplets;

    for (int i = 0; i < nIdx.size(); i++) {
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
        Eigen::Vector4d vecs[3];
        int j = 0;
        for (auto k: mesh_data.FaceNeighbors[data.b(i)]) {
            vecs[j] = mesh_data.Planes.row(k);
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
        for (auto k: mesh_data.FaceNeighbors[data.b(i)]) {
            dists.push_back(dist(j));
            mesh_data.Planes(k, 3) = dist(j);
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
    Eigen::VectorXd newB(b.size() - data.conP.size());
    for (int i = 0; i < b.size(); i++) {
        if (data.distPos[i] < 0) {
            continue;
        }
        newB(data.distPos[i]) = b(i);
        for (int j = 0; j < data.conP.size(); j++) {
            int deletedIndex = data.conP[j];
            newB(data.distPos[i]) -= M.coeff(i, deletedIndex) * dists[j];
        }
    }

    // for (int j = 0; j < data.conP.size(); j++) {
    //     int deletedPoly = data.conP[j];
    //     newB(deletedPoly) = dists[j];
    // }

    int newRows = M.rows() - data.conP.size();
    int newCols = M.cols() - data.conP.size();
    Eigen::SparseMatrix<double> newM(newRows, newCols);
    std::vector<Eigen::Triplet<double> > mTriplets;
    for (int i = 0; i < M.outerSize(); ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it) {
            int row = it.row();
            int col = it.col();
            double value = it.value();
            if (data.distPos[row] >= 0 && data.distPos[col] >= 0) {
                mTriplets.emplace_back(data.distPos[row], data.distPos[col], value);
            }
        }
    }
    newM.setFromTriplets(mTriplets.begin(), mTriplets.end());

    //constraints cover whole mesh
    if (newM.size() == 0) {
        return true;
    }


    // Eigen::VectorXd bestDistances = newM.llt().solve(newB);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
    solver.compute(newM);
    Eigen::VectorXd bestDistances = solver.solve(newB);

    for (int i = 0; i < mesh_data.Planes.rows(); i++) {
        bool skipped = data.distPos[i] < 0;
        // for (int poly = 0; poly < data.conP.size(); poly++) {
        //     if (data.conP[poly] == i) {
        //         skipped = true;
        //     }
        // }
        if (skipped) {
            continue;
        }

        mesh_data.Planes(i, 3) = bestDistances(data.distPos[i]);
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
                              //TODO: einfach constraint berechnung hier rein packen, dann sollte es ja eig. gehen

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
                                  Eigen::Vector4<T> vecs[3];
                                  int j = 0;
                                  for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                                      vecs[j] = element.variables(k);
                                      j++;
                                  }
                                  Eigen::Vector3<T> normal1 = vecs[0].head(3).normalized();
                                  Eigen::Vector3<T> normal2 = vecs[1].head(3).normalized();
                                  Eigen::Vector3<T> normal3 = vecs[2].head(3).normalized();

                                  Eigen::Matrix3<T> m;
                                  m.row(0) = normal1;
                                  m.row(1) = normal2;
                                  m.row(2) = normal3;

                                  Eigen::Vector3<T> dist =
                                          m * bc.row(localConstrainsIndex[i]).transpose();
                                  j = 0;
                                  for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                                      Eigen::Vector4<T> pol = element.variables(k);
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
                                      pol = element.variables(f);
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


TinyAD::ScalarFunction<4, double, long long> getEdgeFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    TinyAD::ScalarFunction<4, double, long long> func = TinyAD::scalar_function<4>(TinyAD::range(data.Polygons.rows()));

    func.add_elements<9>(TinyAD::range(data.edges.size()),
                         [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                             //calculate arap energy
                             using T = TINYAD_SCALAR_TYPE(


                                 element
                             );
                             //TODO: einfach constraint berechnung hier rein packen, dann sollte es ja eig. gehen

                             Eigen::Index e_idx = element.handle;
                             edge e = data.edges[e_idx];
                             std::vector<int> localConstrainsIndex;

                             std::set<int> faces;
                             std::set_union(mesh_data.FaceNeighbors[e.a].begin(), mesh_data.FaceNeighbors[e.a].end(),
                                            mesh_data.FaceNeighbors[e.b].begin(), mesh_data.FaceNeighbors[e.b].end(),
                                            std::inserter(faces, faces.begin()));
                             // faces.insert(mesh_data.VertPolygons[e.b].begin(), mesh_data.VertPolygons[e.b].end());
                             for (int i = 0; i < data.b.size(); i++) {
                                 if (e.a == data.b(i) || e.b == data.b(i)) {
                                     localConstrainsIndex.push_back(i);
                                 }
                                 for (auto p: faces) {
                                     int size = mesh_data.F[i].size();
                                     for (int fi = 0; fi < size; fi++) {
                                         if (mesh_data.F[p][fi] == data.b(i)) {
                                             localConstrainsIndex.push_back(i);
                                         }
                                     }
                                 }
                             }

                             std::map<int, Eigen::Vector4<T> > gons;
                             for (int i = 0; i < localConstrainsIndex.size(); i++) {
                                 Eigen::Vector4<T> vecs[3];
                                 int j = 0;
                                 for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                                     vecs[j] = element.variables(k);
                                     j++;
                                 }
                                 Eigen::Vector3<T> normal1 = vecs[0].head(3).normalized();
                                 Eigen::Vector3<T> normal2 = vecs[1].head(3).normalized();
                                 Eigen::Vector3<T> normal3 = vecs[2].head(3).normalized();

                                 Eigen::Matrix3<T> m;
                                 m.row(0) = normal1;
                                 m.row(1) = normal2;
                                 m.row(2) = normal3;

                                 Eigen::Vector3<T> dist =
                                         m * bc.row(localConstrainsIndex[i]).transpose();
                                 j = 0;
                                 for (auto k: mesh_data.FaceNeighbors[data.b(localConstrainsIndex[i])]) {
                                     Eigen::Vector4<T> pol = element.variables(k);
                                     pol(3) = dist(j);
                                     gons[k] = pol;
                                     j++;
                                 }
                             }

                             std::vector<Eigen::Vector4<T> > polygons;
                             for (auto f: mesh_data.FaceNeighbors[e.a]) {
                                 Eigen::Vector4<T> pol;
                                 if (gons.find(f) != gons.end()) {
                                     pol = gons[f];
                                 } else {
                                     pol = element.variables(f);
                                 }
                                 polygons.push_back(pol);
                             }
                             Eigen::Vector3<T> a = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                             Eigen::Vector3d ogA = data.V.row(e.a);

                             polygons.clear();
                             for (auto f: mesh_data.FaceNeighbors[e.b]) {
                                 Eigen::Vector4<T> pol;
                                 if (gons.find(f) != gons.end()) {
                                     pol = gons[f];
                                 } else {
                                     pol = element.variables(f);
                                 }
                                 polygons.push_back(pol);
                             }
                             Eigen::Vector3<T> b = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                             Eigen::Vector3d ogB = data.V.row(e.b);

                             Eigen::Matrix3d Rot = data.R.block<3, 3>(0, e.rot * 3);
                             // Eigen::MatrixXd V1(points.size() * 3, 3);
                             // Eigen::MatrixX<T> V2(points.size() * 3, 3);
                             //
                             // for (int j = 0; j < size; j++) {
                             //     int next = (j + 1) % size;
                             //     int n1 = mesh_data.Hoods[v_idx][j];
                             //     int n2 = mesh_data.Hoods[v_idx][next];
                             //     Eigen::Vector3d originalNeighbor = data.V.row(n1);
                             //     Eigen::Vector3<T> newNeighbor = points[j];
                             //     Eigen::Vector3d originalNeighbor2 = data.V.row(n2);
                             //     Eigen::Vector3<T> newNeighbor2 = points[next];
                             //     V1.row(j * 3) = data.cotanWeights[v_idx][j * 3 + 2] * (ogVert - originalNeighbor);
                             //     V2.row(j * 3) = data.cotanWeights[v_idx][j * 3 + 2] * (vert - newNeighbor);
                             //
                             //     V1.row(j * 3 + 1) =
                             //             data.cotanWeights[v_idx][j * 3 + 1] * (ogVert - originalNeighbor2);
                             //     V2.row(j * 3 + 1) = data.cotanWeights[v_idx][j * 3 + 1] * (vert - newNeighbor2);
                             //
                             //     V1.row(j * 3 + 2) =
                             //             data.cotanWeights[v_idx][j * 3] * (originalNeighbor - originalNeighbor2);
                             //     V2.row(j * 3 + 2) = data.cotanWeights[v_idx][j * 3] * (newNeighbor - newNeighbor2);
                             // }
                             //
                             // Eigen::Matrix3<T> Rot = getRotation<T>(V1, V2);

                             // Eigen::Matrix3d Rot = data.R.block<3, 3>(0, v_idx * 3);

                             T returnValue = 0;
                             Eigen::Vector3d v = ogB - ogA;
                             Eigen::Vector3<T> tv = b - a;

                             returnValue += e.w * (tv - Rot * v).squaredNorm();
                             return returnValue;
                         });
    return func;
}
