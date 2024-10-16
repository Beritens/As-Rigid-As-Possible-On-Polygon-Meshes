//
// Created by ben on 25.09.24.
//

#include "../include/face_arap.h"


#include <cassert>

#include <iostream>

template<typename Scalar>
using MatrixXX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

void insertInFaceL(Eigen::MatrixXd &L, int v1, int v2, double cot) {
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

bool face_arap_precomputation(
    poly_mesh_data &mesh_data,
    face_arap_data &data,
    const Eigen::VectorXi &b) {
    using namespace std;
    using namespace Eigen;
    // number of vertices
    const int n = mesh_data.V.rows();


    data.cotanWeights.clear();
    data.triangles.clear();
    data.cotanWeights.resize(mesh_data.F.rows());
    data.triangles.resize(mesh_data.F.rows());

    for (int i = 0; i < mesh_data.F.rows(); i++) {
        int size = faceSize(mesh_data.F.row(i));
        std::vector<Eigen::Vector3d> vecs;
        for (int j = 0; j < size; j++) {
            vecs.push_back(mesh_data.V.row(mesh_data.F(i, j)));
        }
        std::vector<std::vector<int> > triangles = calculateTriangle(vecs, mesh_data.Polygons.row(i).head(3));
        std::vector<std::vector<int> > tris;
        for (auto tri: triangles) {
            std::vector<int> indexes;
            for (int j = 0; j < 3; j++) {
                if (tri[j] < 0 || tri[j] >= size) {
                    continue;
                }
                indexes.push_back(mesh_data.F(i, tri[j]));
            }
            tris.push_back(indexes);
        }
        data.triangles[i] = tris;
    }

    for (int i = 0; i < data.triangles.size(); i++) {
        data.cotanWeights[i].resize(data.triangles[i].size());
        for (int j = 0; j < data.triangles[i].size(); j++) {
            std::vector<int> tri = data.triangles[i][j];
            Eigen::Vector3d v_a = mesh_data.V.row(tri[0]);
            Eigen::Vector3d v_b = mesh_data.V.row(tri[1]);
            Eigen::Vector3d v_c = mesh_data.V.row(tri[2]);

            double angleA = getAngle(v_b - v_a, v_c - v_a);
            double angleB = getAngle(v_a - v_b, v_c - v_b);
            double angleC = getAngle(v_a - v_c, v_b - v_c);
            // data.cotanWeights[i].push_back(1.0);
            // data.cotanWeights[i].push_back(1.0);
            // data.cotanWeights[i].push_back(1.0);

            data.cotanWeights[i][j].push_back(1.0 / tan(angleA));
            data.cotanWeights[i][j].push_back(1.0 / tan(angleB));
            data.cotanWeights[i][j].push_back(1.0 / tan(angleC));
        }
    }


    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * mesh_data.V.rows(),
                                              3 * mesh_data.V.rows());


    for (int i = 0; i < data.triangles.size(); i++) {
        for (int j = 0; j < data.triangles[i].size(); j++) {
            int a = data.triangles[i][j][0];
            int b = data.triangles[i][j][1];
            int c = data.triangles[i][j][2];
            insertInFaceL(L, a, b, data.cotanWeights[i][j][2]);
            insertInFaceL(L, a, c, data.cotanWeights[i][j][1]);
            insertInFaceL(L, b, c, data.cotanWeights[i][j][0]);
        }
    }

    data.L = L;
    std::cout << "L" << std::endl;
    std::cout << L << std::endl;
    data.Polygons = mesh_data.Polygons;
    data.b = b;
    data.V = mesh_data.V;
    data.R = Eigen::MatrixXd(3, mesh_data.F.rows() * 3);
    return true;
}

void getFaceRotations(poly_mesh_data &mesh_data, face_arap_data &data) {
    for (int i = 0; i < data.triangles.size(); i++) {
        Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(data.triangles[i].size() * 3, 3);
        Eigen::MatrixXd V2 = Eigen::MatrixXd::Zero(data.triangles[i].size() * 3, 3);
        for (int j = 0; j < data.triangles[i].size(); j++) {
            std::vector<int> tri = data.triangles[i][j];
            Eigen::Vector3d v_a = mesh_data.V.row(tri[0]);
            Eigen::Vector3d v_b = mesh_data.V.row(tri[1]);
            Eigen::Vector3d v_c = mesh_data.V.row(tri[2]);
            Eigen::Vector3d ov_a = data.V.row(tri[0]);
            Eigen::Vector3d ov_b = data.V.row(tri[1]);
            Eigen::Vector3d ov_c = data.V.row(tri[2]);
            V1.row(j * 3) = data.cotanWeights[i][j][2] * (ov_a - ov_b);
            V2.row(j * 3) = data.cotanWeights[i][j][2] * (v_a - v_b);

            V1.row(j * 3 + 1) = data.cotanWeights[i][j][1] * (ov_a - ov_c);
            V2.row(j * 3 + 1) = data.cotanWeights[i][j][1] * (v_a - v_c);

            V1.row(j * 3 + 2) = data.cotanWeights[i][j][0] * (ov_c - ov_b);
            V2.row(j * 3 + 2) = data.cotanWeights[i][j][0] * (v_c - v_b);
        }

        Eigen::Matrix3d rot = getRotation(V1, V2);
        data.R.block<3, 3>(0, i * 3) = rot;
    }
}

void insertInFaceB(Eigen::VectorXd &b, int v1, int v2, double cot, const Eigen::MatrixXd V, const Eigen::Matrix3d rot) {
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


bool global_face_distance_step(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    face_arap_data &data) {
    using namespace Eigen;
    using namespace std;
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const int n = data.V.rows();
    std::vector<Eigen::MatrixXd> invNs;
    std::vector<std::vector<int> > nIdx;

    Eigen::VectorXd nV(data.V.rows() * 3);

    for (int i = 0; i < data.V.rows(); i++) {
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

    Eigen::MatrixXd NInv = Eigen::MatrixXd::Zero(data.V.rows() * 3, mesh_data.Polygons.rows());
    for (int i = 0; i < nIdx.size(); i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < nIdx[i].size(); k++) {
                NInv(i * 3 + j, nIdx[i][k]) += invNs[i](j, k);
            }
        }
    }
    Eigen::MatrixXd NInvT = NInv.transpose();


    // Eigen::VectorXd nV = NInv * d;
    //test


    Eigen::VectorXd d(mesh_data.Polygons.rows());
    for (int i = 0; i < d.size(); i++) {
        d(i) = mesh_data.Polygons(i, 3);
    }


    Eigen::MatrixXd M = NInvT * data.L * NInv;

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


    Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * data.V.rows());

    // for (int i = 0; i < mesh_data.V.rows(); i++) {
    //     std::vector val = mesh_data.Hoods[i];
    //     int size = val.size();
    //
    //     Eigen::Matrix3d rot = data.R.block<3,
    //
    //
    //         3>(0, i * 3);
    //
    //     for (int j = 0; j < size; j++) {
    //         int next = (j + 1) % size;
    //         int n1 = mesh_data.Hoods[i][j];
    //         int n2 = mesh_data.Hoods[i][next];
    //         //anders probieren. einfach laplacian benutzen und dann direkt edge rein addieren I guess
    //         insertInFaceB(b, i, n1, data.cotanWeights[i][j * 3 + 2] * 1, data.V, rot);
    //         insertInFaceB(b, i, n2, data.cotanWeights[i][j * 3 + 1] * 1, data.V, rot);
    //         insertInFaceB(b, n2, n1, data.cotanWeights[i][j * 3] * 6, data.V, rot);
    //     }
    // }

    for (int i = 0; i < data.triangles.size(); i++) {
        Eigen::Matrix3d rot = data.R.block<3, 3>(0, i * 3);
        for (int j = 0; j < data.triangles[i].size(); j++) {
            std::vector<int> tri = data.triangles[i][j];
            insertInFaceB(b, tri[0], tri[1], data.cotanWeights[i][j][2], data.V, rot);
            insertInFaceB(b, tri[0], tri[2], data.cotanWeights[i][j][1], data.V, rot);
            insertInFaceB(b, tri[2], tri[1], data.cotanWeights[i][j][0], data.V, rot);
        }
    }

    Eigen::VectorXd newB = NInvT * b;

    // std::cout << "should be 0" << std::endl;
    // std::cout << M * d - b << std::endl;
    for (int j = 0; j < gons.size(); j++) {
        int deletedPoly = gons[j];
        newB(deletedPoly) = dists[j];
        // rightSide(0) -= M(3 * i, deletedPoly) * dists[j];
        // rightSide(1) -= M(3 * i + 1, deletedPoly) * dists[j];
        // rightSide(2) -= M(3 * i + 2, deletedPoly) * dists[j];
    }

    // for (int i = 0; i < newB.rows(); i++) {
    //     // Eigen::Matrix3d rot = R.block<3, 3>(0, i * 3);
    //
    //     // for (auto v: mesh_data.Hoods[i]) {
    //     //     rightSide -= rot * (data.V.row(v) - data.V.row(i)).transpose();
    //     // }
    //     // for (int j = 0; j < gons.size(); j++) {
    //     //     int deletedPoly = gons[j];
    //     //     newB(i) -= M(i, deletedPoly) * dists[j];
    //     //     // rightSide(0) -= M(3 * i, deletedPoly) * dists[j];
    //     //     // rightSide(1) -= M(3 * i + 1, deletedPoly) * dists[j];
    //     //     // rightSide(2) -= M(3 * i + 2, deletedPoly) * dists[j];
    //     // }
    //     // b(i * 3 + 1) += rightSide(1);
    //     // b(i * 3 + 2) += rightSide(2);
    // }


    Eigen::MatrixXd newM(M.rows(), M.cols());
    for (int i = 0; i < M.rows(); i++) {
        bool deltedRow = false;
        for (int poly = 0; poly < gons.size(); poly++) {
            if (gons[poly] == i) {
                deltedRow = true;
            }
        }
        if (deltedRow) {
            newM(i, i) = 1;
        }
        for (int j = 0; j < M.cols(); j++) {
            if (deltedRow) {
                if (j != i) {
                    newM(i, j) = 0;
                }
            } else {
                newM(i, j) = M(i, j);
            }
        }
    }


    //constraints cover whole mesh
    if (newM.size() == 0) {
        return true;
    }
    std::cout << "test" << std::endl;
    std::cout << newM << std::endl;
    std::cout << newB << std::endl;

    Eigen::VectorXd bestDistances = newM.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(newB);
    // std::cout << "should be zero" << std::endl;
    // std::cout << newM * bestDistances - b << std::endl;
    // std::cout << "R" << std::endl;
    // std::cout << data.R << std::endl;
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

        mesh_data.Polygons(i, 3) = bestDistances(i);
    }

    return true;
}

TinyAD::ScalarFunction<4, double, long> getFaceFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    face_arap_data &data) {
    auto func = TinyAD::scalar_function<4>(TinyAD::range(data.Polygons.rows()));

    func.add_elements<10>(TinyAD::range(data.triangles.size()),
                          [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                              //calculate arap energy
                              using T = TINYAD_SCALAR_TYPE(


                                  element
                              );
                              //TODO: einfach constraint berechnung hier rein packen, dann sollte es ja eig. gehen

                              Eigen::Index f_idx = element.handle;
                              std::vector<int> localConstrainsIndex;


                              int size = faceSize(mesh_data.F.row(f_idx));
                              for (int i = 0; i < size; i++) {
                                  for (auto p: mesh_data.VertPolygons[mesh_data.F(f_idx, i)]) {
                                      int size2 = faceSize(mesh_data.F.row(p));
                                      for (int fi = 0; fi < size2; fi++) {
                                          for (int j = 0; j < data.b.size(); j++) {
                                              if (mesh_data.F(p, fi) == data.b(j)) {
                                                  localConstrainsIndex.push_back(j);
                                              }
                                          }
                                      }
                                  }
                              }


                              std::map<int, Eigen::Vector4<T> > gons;
                              for (int i = 0; i < localConstrainsIndex.size(); i++) {
                                  Eigen::Vector4<T> vecs[3];
                                  int j = 0;
                                  for (auto k: mesh_data.VertPolygons[data.b(localConstrainsIndex[i])]) {
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
                                  for (auto k: mesh_data.VertPolygons[data.b(localConstrainsIndex[i])]) {
                                      Eigen::Vector4<T> pol = element.variables(k);
                                      pol(3) = dist(j);
                                      gons[k] = pol;
                                      j++;
                                  }
                              }


                              std::map<int, Eigen::Vector3<T> > points;

                              for (int i = 0; i < size; i++) {
                                  std::vector<Eigen::Vector4<T> > neighPolygons;
                                  for (auto f: mesh_data.VertPolygons[mesh_data.F(f_idx, i)]) {
                                      Eigen::Vector4<T> pol;
                                      if (gons.find(f) != gons.end()) {
                                          pol = gons[f];
                                      } else {
                                          pol = element.variables(f);
                                      }
                                      // Eigen::Vector3<T> normal = element.variables(f);
                                      // pol.head(3) = normal;
                                      // pol(3) = mesh_data.Polygons(f, 3);
                                      neighPolygons.push_back(pol);
                                  }
                                  Eigen::Vector3<T> neighborVert = getPoint<T>(
                                      neighPolygons[0], neighPolygons[1], neighPolygons[2]);
                                  points[mesh_data.F(f_idx, i)] = neighborVert;
                              }

                              Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(data.triangles[f_idx].size() * 3, 3);
                              Eigen::MatrixX<T> V2 = Eigen::MatrixXd::Zero(data.triangles[f_idx].size() * 3, 3);
                              for (int j = 0; j < data.triangles[f_idx].size(); j++) {
                                  std::vector<int> tri = data.triangles[f_idx][j];
                                  Eigen::Vector3<T> v_a = points[tri[0]];
                                  Eigen::Vector3<T> v_b = points[tri[1]];
                                  Eigen::Vector3<T> v_c = points[tri[2]];
                                  Eigen::Vector3d ov_a = data.V.row(tri[0]);
                                  Eigen::Vector3d ov_b = data.V.row(tri[1]);
                                  Eigen::Vector3d ov_c = data.V.row(tri[2]);
                                  V1.row(j * 3) = data.cotanWeights[f_idx][j][2] * (ov_a - ov_b);
                                  V2.row(j * 3) = data.cotanWeights[f_idx][j][2] * (v_a - v_b);

                                  V1.row(j * 3 + 1) = data.cotanWeights[f_idx][j][1] * (ov_a - ov_c);
                                  V2.row(j * 3 + 1) = data.cotanWeights[f_idx][j][1] * (v_a - v_c);

                                  V1.row(j * 3 + 2) = data.cotanWeights[f_idx][j][0] * (ov_c - ov_b);
                                  V2.row(j * 3 + 2) = data.cotanWeights[f_idx][j][0] * (v_c - v_b);
                              }

                              Eigen::Matrix3<T> Rot = getRotation<T>(V1, V2);

                              T returnValue = 0;
                              // for (int j = 0; j < size; j++) {
                              //     int next = (j + 1) % size;
                              //     int n1 = mesh_data.Hoods[v_idx][j];
                              //     int n2 = mesh_data.Hoods[v_idx][next];
                              //     Eigen::Vector3d ogNeighbor = data.V.row(n1);
                              //     Eigen::Vector3d ogNeighbor2 = data.V.row(n2);
                              //     Eigen::Vector3d v = ogNeighbor - ogVert;
                              //     Eigen::Vector3<T> tv = points[j] - vert;
                              //
                              //     Eigen::Vector3d v2 = ogNeighbor2 - ogVert;
                              //     Eigen::Vector3<T> tv2 = points[next] - vert;
                              //
                              //     Eigen::Vector3d v3 = ogNeighbor2 - ogNeighbor;
                              //     Eigen::Vector3<T> tv3 = points[next] - points[j];
                              //
                              //     returnValue += data.cotanWeights[v_idx][j * 3 + 2] * (tv - Rot * v).squaredNorm();
                              //     returnValue += data.cotanWeights[v_idx][j * 3 + 1] * (tv2 - Rot * v2).
                              //             squaredNorm();
                              //     returnValue += data.cotanWeights[v_idx][j * 3] * (tv3 - Rot * v3).squaredNorm();
                              // }
                              for (int j = 0; j < data.triangles[f_idx].size(); j++) {
                                  std::vector<int> tri = data.triangles[f_idx][j];
                                  Eigen::Vector3<T> v_a = points[tri[0]];
                                  Eigen::Vector3<T> v_b = points[tri[1]];
                                  Eigen::Vector3<T> v_c = points[tri[2]];
                                  Eigen::Vector3d ov_a = data.V.row(tri[0]);
                                  Eigen::Vector3d ov_b = data.V.row(tri[1]);
                                  Eigen::Vector3d ov_c = data.V.row(tri[2]);
                                  returnValue += data.cotanWeights[f_idx][j][2] * ((v_b - v_a) - Rot * (ov_b - ov_a)).
                                          squaredNorm();
                                  returnValue += data.cotanWeights[f_idx][j][1] * ((v_c - v_a) - Rot * (ov_c - ov_a)).
                                          squaredNorm();
                                  returnValue += data.cotanWeights[f_idx][j][0] * ((v_b - v_c) - Rot * (ov_b - ov_c)).
                                          squaredNorm();
                              }

                              return returnValue;
                          });
    return func;
}
