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
    const int n = mesh_data.originalV.rows();


    data.cotanWeights.clear();
    data.triangles.clear();
    data.cotanWeights.resize(mesh_data.F.size());
    data.triangles.resize(mesh_data.F.size());
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(3 * (mesh_data.originalV.rows() + mesh_data.F.size()),
                                                  3 * mesh_data.originalV.rows());

    data.simpleB = Eigen::MatrixXd::Identity((mesh_data.originalV.rows() + mesh_data.F.size()),
                                             mesh_data.originalV.rows());

    Eigen::MatrixXd virtualVerts(mesh_data.F.size(), 3);
    //finding virtual point
    // took this from here: https://www.cs.jhu.edu/~misha/MyPapers/EUROG20.pdf

    for (int i = 0; i < mesh_data.F.size(); i++) {
        int size = mesh_data.F[i].size();
        Eigen::Vector3d virtVert = Vector3d::Zero();
        Eigen::MatrixXd A(size + 1, size);
        Eigen::VectorXd vb(size + 1);

        for (int j = 0; j < size; j++) {
            double sumB = 0;
            for (int k = 0; k < size; k++) {
                double sumA = 0;
                for (int l = 0; l < size; l++) {
                    Eigen::Vector3d xk = mesh_data.originalV.row(mesh_data.F[i][k]);
                    Eigen::Vector3d xj = mesh_data.originalV.row(mesh_data.F[i][j]);
                    Eigen::Vector3d xjplus = mesh_data.originalV.row(mesh_data.F[i][(j + 1) % size]);
                    Eigen::Vector3d xl = mesh_data.originalV.row(mesh_data.F[i][l]);
                    Eigen::Vector3d xlplus = mesh_data.originalV.row(mesh_data.F[i][(l + 1) % size]);
                    sumA += (xk.cross(xlplus - xl)).dot(xj.cross(xlplus - xl));
                }
                A(j, k) = 2 * sumA;
            }
            for (int l = 0; l < size; l++) {
                Eigen::Vector3d xj = mesh_data.originalV.row(mesh_data.F[i][j]);
                Eigen::Vector3d xjplus = mesh_data.originalV.row(mesh_data.F[i][(j + 1) % size]);
                Eigen::Vector3d xl = mesh_data.originalV.row(mesh_data.F[i][l]);
                Eigen::Vector3d xlplus = mesh_data.originalV.row(mesh_data.F[i][(l + 1) % size]);
                sumB += (xj.cross(xlplus - xl)).dot((xlplus - xl).cross(xl));
            }
            vb(j) = 2 * sumB;


            // virtVert += mesh_data.originalV.row(mesh_data.F(i, j)) / (double) size;
            // data.simpleB((mesh_data.originalV.rows() + i), (mesh_data.F(i, j))) = 1.0 / (double) size;
            // data.B(3 * (mesh_data.originalV.rows() + i), 3 * (mesh_data.F(i, j))) = 1.0 / (double) size;
            // data.B(3 * (mesh_data.originalV.rows() + i) + 1, 3 * (mesh_data.F(i, j)) + 1) = 1.0 / (double) size;
            // data.B(3 * (mesh_data.originalV.rows() + i) + 2, 3 * (mesh_data.F(i, j)) + 2) = 1.0 / (double) size;
            A(size, j) = 1;
        }
        vb(size) = -1;

        Eigen::VectorXd w = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-vb);
        // Eigen::VectorXd w = Eigen::VectorXd(size) / (double) size;

        for (int j = 0; j < size; j++) {
            virtVert += mesh_data.originalV.row(mesh_data.F[i][j]) * w(j);
            data.simpleB((mesh_data.originalV.rows() + i), (mesh_data.F[i][j])) = w(j);
            B(3 * (mesh_data.originalV.rows() + i), 3 * (mesh_data.F[i][j])) = w(j);
            B(3 * (mesh_data.originalV.rows() + i) + 1, 3 * (mesh_data.F[i][j]) + 1) = w(j);
            B(3 * (mesh_data.originalV.rows() + i) + 2, 3 * (mesh_data.F[i][j]) + 2) = w(j);
        }

        virtualVerts.row(i) = virtVert;
    }


    data.V = Eigen::MatrixXd(mesh_data.originalV.rows() + virtualVerts.rows(), 3);
    data.V << mesh_data.originalV, virtualVerts;

    for (int i = 0; i < mesh_data.F.size(); i++) {
        std::vector<std::vector<int> > tris;
        int size = mesh_data.F[i].size();
        for (int j = 0; j < size; j++) {
            int next = (j + 1) % size;
            std::vector<int> tri;
            tri.push_back(mesh_data.originalV.rows() + i);
            tri.push_back(mesh_data.F[i][j]);
            tri.push_back(mesh_data.F[i][next]);
            tris.push_back(tri);
        }
        data.triangles[i] = tris;
    }

    // for (int i = 0; i < mesh_data.F.rows(); i++) {
    //     int size = faceSize(mesh_data.F.row(i));
    //     std::vector<Eigen::Vector3d> vecs;
    //     for (int j = 0; j < size; j++) {
    //         vecs.push_back(mesh_data.V.row(mesh_data.F(i, j)));
    //     }
    //     std::vector<std::vector<int> > triangles = calculateTriangle(vecs, mesh_data.Polygons.row(i).head(3));
    //     std::vector<std::vector<int> > tris;
    //     for (auto tri: triangles) {
    //         std::vector<int> indexes;
    //         for (int j = 0; j < 3; j++) {
    //             if (tri[j] < 0 || tri[j] >= size) {
    //                 continue;
    //             }
    //             indexes.push_back(mesh_data.F(i, tri[j]));
    //         }
    //         tris.push_back(indexes);
    //     }
    //     data.triangles[i] = tris;
    // }

    for (int i = 0; i < data.triangles.size(); i++) {
        data.cotanWeights[i].resize(data.triangles[i].size());
        for (int j = 0; j < data.triangles[i].size(); j++) {
            std::vector<int> tri = data.triangles[i][j];
            Eigen::Vector3d v_a = data.V.row(tri[0]);
            Eigen::Vector3d v_b = data.V.row(tri[1]);
            Eigen::Vector3d v_c = data.V.row(tri[2]);

            double angleA = getAngle(v_b - v_a, v_c - v_a);
            double angleB = getAngle(v_a - v_b, v_c - v_b);
            double angleC = getAngle(v_a - v_c, v_b - v_c);
            // data.cotanWeights[i].push_back(1.0);
            // data.cotanWeights[i].push_back(1.0);
            // data.cotanWeights[i].push_back(1.0);

            data.cotanWeights[i][j].push_back(1.0 / max(tan(angleA), 0.001));
            data.cotanWeights[i][j].push_back(1.0 / max(tan(angleB), 0.001));
            data.cotanWeights[i][j].push_back(1.0 / max(tan(angleC), 0.001));
        }
    }


    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * data.V.rows(),
                                              3 * data.V.rows());


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


    data.L = L.sparseView(1e-9);
    data.B = B.sparseView(1e-9);

    data.Polygons = mesh_data.Planes;
    data.b = b;
    data.R = Eigen::MatrixXd(3, mesh_data.F.size() * 3);
    data.Bt = data.B.transpose();

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

void getFaceRotations(poly_mesh_data &mesh_data, face_arap_data &data) {
    Eigen::MatrixXd withVirtVerts = data.simpleB * mesh_data.V;
    for (int i = 0; i < data.triangles.size(); i++) {
        Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(data.triangles[i].size() * 3, 3);
        Eigen::MatrixXd V2 = Eigen::MatrixXd::Zero(data.triangles[i].size() * 3, 3);
        for (int j = 0; j < data.triangles[i].size(); j++) {
            std::vector<int> tri = data.triangles[i][j];
            Eigen::Vector3d v_a = withVirtVerts.row(tri[0]);
            Eigen::Vector3d v_b = withVirtVerts.row(tri[1]);
            Eigen::Vector3d v_c = withVirtVerts.row(tri[2]);
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


    for (int i = 0; i < mesh_data.originalV.rows(); i++) {
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

    int rows = mesh_data.originalV.rows() * 3;
    int cols = mesh_data.Planes.rows();
    Eigen::SparseMatrix<double> NInv(rows, cols);
    std::vector<Eigen::Triplet<double> > triplets;

    for (int i = 0; i < nIdx.size(); i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < nIdx[i].size(); k++) {
                triplets.emplace_back(i * 3 + j, nIdx[i][k], invNs[i](j, k));
            }
        }
    }
    NInv.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SparseMatrix<double> NInvT = NInv.transpose();


    // Eigen::VectorXd nV = NInv * d;
    //test


    Eigen::VectorXd d(mesh_data.Planes.rows());
    for (int i = 0; i < d.size(); i++) {
        d(i) = mesh_data.Planes(i, 3);
    }


    Eigen::SparseMatrix<double> M = NInvT * data.Bt * data.L * data.B * NInv;

    std::vector<int> gons;
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
            gons.push_back(k);
            dists.push_back(dist(j));
            mesh_data.Planes(k, 3) = dist(j);
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
    assert(!b.hasNaN());
    b = NInvT * data.Bt * b;

    // std::cout << "should be 0" << std::endl;
    // std::cout << M * d - b << std::endl;

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

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
    solver.compute(newM);
    Eigen::VectorXd bestDistances = solver.solve(newB);
    assert(!bestDistances.hasNaN());

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

TinyAD::ScalarFunction<4, double, long long> getFaceFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    face_arap_data &data) {
    TinyAD::ScalarFunction<4, double, long long> func = TinyAD::scalar_function<4>(TinyAD::range(data.Polygons.rows()));

    func.add_elements<10>(TinyAD::range(data.triangles.size()),
                          [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                              //calculate arap energy
                              using T = TINYAD_SCALAR_TYPE(


                                  element
                              );
                              //TODO: einfach constraint berechnung hier rein packen, dann sollte es ja eig. gehen

                              Eigen::Index f_idx = element.handle;
                              std::vector<int> localConstrainsIndex;


                              int size = mesh_data.F[f_idx].size();
                              for (int i = 0; i < size; i++) {
                                  for (auto p: mesh_data.FaceNeighbors[mesh_data.F[f_idx][i]]) {
                                      int size2 = mesh_data.F[p].size();
                                      for (int fi = 0; fi < size2; fi++) {
                                          for (int j = 0; j < data.b.size(); j++) {
                                              if (mesh_data.F[p][fi] == data.b(j)) {
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


                              std::map<int, Eigen::Vector3<T> > points;
                              Eigen::Vector3<T> virtVert = Eigen::Vector3d::Zero();

                              for (int i = 0; i < size; i++) {
                                  std::vector<Eigen::Vector4<T> > neighPolygons;
                                  for (auto f: mesh_data.FaceNeighbors[mesh_data.F[f_idx][i]]) {
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
                                  points[mesh_data.F[f_idx][i]] = neighborVert;
                                  virtVert += neighborVert * data.B.coeff(3 * (f_idx + mesh_data.originalV.rows()),
                                                                          3 * mesh_data.F[f_idx][i]);
                              }
                              points[mesh_data.V.rows() + f_idx] = virtVert;

                              // Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(data.triangles[f_idx].size() * 3, 3);
                              // Eigen::MatrixX<T> V2 = Eigen::MatrixXd::Zero(data.triangles[f_idx].size() * 3, 3);
                              // for (int j = 0; j < data.triangles[f_idx].size(); j++) {
                              //     std::vector<int> tri = data.triangles[f_idx][j];
                              //     Eigen::Vector3<T> v_a = points[tri[0]];
                              //     Eigen::Vector3<T> v_b = points[tri[1]];
                              //     Eigen::Vector3<T> v_c = points[tri[2]];
                              //     Eigen::Vector3d ov_a = data.V.row(tri[0]);
                              //     Eigen::Vector3d ov_b = data.V.row(tri[1]);
                              //     Eigen::Vector3d ov_c = data.V.row(tri[2]);
                              //     V1.row(j * 3) = data.cotanWeights[f_idx][j][2] * (ov_a - ov_b);
                              //     V2.row(j * 3) = data.cotanWeights[f_idx][j][2] * (v_a - v_b);
                              //
                              //     V1.row(j * 3 + 1) = data.cotanWeights[f_idx][j][1] * (ov_a - ov_c);
                              //     V2.row(j * 3 + 1) = data.cotanWeights[f_idx][j][1] * (v_a - v_c);
                              //
                              //     V1.row(j * 3 + 2) = data.cotanWeights[f_idx][j][0] * (ov_c - ov_b);
                              //     V2.row(j * 3 + 2) = data.cotanWeights[f_idx][j][0] * (v_c - v_b);
                              // }
                              //
                              // Eigen::Matrix3<T> Rot = getRotation<T>(V1, V2);


                              Eigen::Matrix3d Rot = data.R.block<3, 3>(0, f_idx * 3);

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
