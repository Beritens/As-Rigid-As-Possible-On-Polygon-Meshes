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
    const int n = mesh_data.V.rows();
    data.positions.clear();

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


    data.cotanWeights.resize(mesh_data.V.rows());

    for (int i = 0; i < mesh_data.V.rows(); i++) {
        for (int j = 0; j < mesh_data.Hoods[i].size(); j++) {
            int next = (j + 1) % mesh_data.Hoods[i].size();
            Eigen::Vector3d v_a = mesh_data.V.row(i);
            Eigen::Vector3d v_b = mesh_data.V.row(mesh_data.Hoods[i][j]);
            Eigen::Vector3d v_c = mesh_data.V.row(mesh_data.Hoods[i][next]);

            double angleA = getAngle(v_b - v_a, v_c - v_a);
            double angleB = getAngle(v_b - v_c, v_a - v_c);
            double angleC = getAngle(v_c - v_b, v_c - v_b);

            data.cotanWeights[i].push_back(1.0 / tan(angleA));
            data.cotanWeights[i].push_back(1.0 / tan(angleB));
            data.cotanWeights[i].push_back(1.0 / tan(angleC));
        }
    }

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3 * mesh_data.V.rows(),
                                              3 * mesh_data.V.rows());
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        std::vector val = mesh_data.Hoods[i];
        int size = val.size();

        for (int j = 0; j < size; j++) {
            int next = (j + 1) % size;
            int n1 = mesh_data.Hoods[i][j];
            int n2 = mesh_data.Hoods[i][next];
            insertInL(L, i, n1, data.cotanWeights[i][j * 3 + 2]);
            insertInL(L, i, n2, data.cotanWeights[i][j * 3 + 1]);
            insertInL(L, n1, n2, data.cotanWeights[i][j * 3]);
        }
    }

    data.L = L;
    std::cout << "L" << std::endl;
    std::cout << L << std::endl;
    data.Polygons = mesh_data.Polygons;
    data.b = b;
    data.V = mesh_data.V;
    data.R = Eigen::MatrixXd(3, mesh_data.V.rows() * 3);
    return true;
}

void getRotations(poly_mesh_data &mesh_data, plane_arap_data &data) {
    for (int i = 0; i < data.V.rows(); i++) {
        Eigen::MatrixXd V1 = Eigen::MatrixXd::Zero(mesh_data.Hoods[i].size() * 3, 3);
        Eigen::MatrixXd V2 = Eigen::MatrixXd::Zero(mesh_data.Hoods[i].size() * 3, 3);

        Eigen::Vector3d originalVert = data.V.row(i);
        Eigen::Vector3d newVert = mesh_data.V.row(i);

        int size = mesh_data.Hoods[i].size();

        for (int j = 0; j < size; j++) {
            int next = (j + 1) % size;
            int n1 = mesh_data.Hoods[i][j];
            int n2 = mesh_data.Hoods[i][next];
            Eigen::Vector3d originalNeighbor = data.V.row(n1);
            Eigen::Vector3d newNeighbor = mesh_data.V.row(n1);
            Eigen::Vector3d originalNeighbor2 = data.V.row(n2);
            Eigen::Vector3d newNeighbor2 = mesh_data.V.row(n2);
            V1.row(j * 3) = (originalVert - originalNeighbor);
            V2.row(j * 3) = (newVert - newNeighbor);

            V1.row(j * 3 + 1) = (originalVert - originalNeighbor2);
            V2.row(j * 3 + 1) = (newVert - newNeighbor2);

            V1.row(j * 3 + 2) = (originalNeighbor - originalNeighbor2);
            V2.row(j * 3 + 2) = (newNeighbor - newNeighbor2);
        }


        Eigen::Matrix3d s = V2.transpose() * V1;
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


    // Eigen::VectorXd nV = NInv * d;
    //test


    Eigen::VectorXd d(mesh_data.Polygons.rows());
    for (int i = 0; i < d.size(); i++) {
        d(i) = mesh_data.Polygons(i, 3);
    }


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


    Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * data.V.rows());

    for (int i = 0; i < mesh_data.V.rows(); i++) {
        std::vector val = mesh_data.Hoods[i];
        int size = val.size();

        Eigen::Matrix3d rot = data.R.block<3, 3>(0, i * 3);

        for (int j = 0; j < size; j++) {
            int next = (j + 1) % size;
            int n1 = mesh_data.Hoods[i][j];
            int n2 = mesh_data.Hoods[i][next];
            insertInB(b, i, n1, data.cotanWeights[i][j * 3 + 2], data.V, rot);
            insertInB(b, i, n2, data.cotanWeights[i][j * 3 + 1], data.V, rot);
            insertInB(b, n1, n2, data.cotanWeights[i][j * 3], data.V, rot);
        }
    }

    int row = 0;
    for (int i = 0; i < data.V.rows(); i++) {
        // Eigen::Matrix3d rot = R.block<3, 3>(0, i * 3);
        Eigen::Vector3d rightSide = Eigen::Vector3d::Zero();

        // for (auto v: mesh_data.Hoods[i]) {
        //     rightSide -= rot * (data.V.row(v) - data.V.row(i)).transpose();
        // }
        for (int j = 0; j < gons.size(); j++) {
            int deletedPoly = gons[j];
            rightSide(0) -= M(3 * i, deletedPoly) * dists[j];
            rightSide(1) -= M(3 * i + 1, deletedPoly) * dists[j];
            rightSide(2) -= M(3 * i + 2, deletedPoly) * dists[j];
        }
        b(row * 3) += rightSide(0);
        b(row * 3 + 1) += rightSide(1);
        b(row * 3 + 2) += rightSide(2);
        row++;
    }

    Eigen::MatrixXd newM(M.rows(), M.cols() - dists.size());
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
            int x = j;
            newM(x, y) = M(j, i);
        }
        y++;
    }

    //constraints cover whole mesh
    if (newM.size() == 0) {
        return true;
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

TinyAD::ScalarFunction<4, double, long> getFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    auto func = TinyAD::scalar_function<4>(TinyAD::range(data.Polygons.rows()));

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
                                  for (auto v: mesh_data.Hoods[v_idx]) {
                                      for (auto p: mesh_data.VertPolygons[v]) {
                                          int size = faceSize(mesh_data.F.row(p));
                                          for (int fi = 0; fi < size; fi++) {
                                              if (mesh_data.F(p, fi) == data.b(i)) {
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

                              std::vector<Eigen::Vector4<T> > polygons;
                              for (auto f: mesh_data.VertPolygons[v_idx]) {
                                  Eigen::Vector4<T> pol;
                                  pol = element.variables(f);
                                  if (gons.find(f) != gons.end()) {
                                      pol = gons[f];
                                  }
                                  // Eigen::Vector3<T> normal = element.variables(f);
                                  // pol.head(3) = normal;
                                  // pol(3) = mesh_data.Polygons(f, 3);
                                  polygons.push_back(pol);
                              }
                              Eigen::Vector3<T> vert = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                              Eigen::Vector3d ogVert = data.V.row(v_idx);
                              std::vector<Eigen::Vector3<T> > points;

                              for (auto neighbor: mesh_data.Hoods[v_idx]) {
                                  std::vector<Eigen::Vector4<T> > neighPolygons;
                                  for (auto f: mesh_data.VertPolygons[neighbor]) {
                                      Eigen::Vector4<T> pol;
                                      pol = element.variables(f);
                                      if (gons.find(f) != gons.end()) {
                                          pol = gons[f];
                                      }
                                      // Eigen::Vector3<T> normal = element.variables(f);
                                      // pol.head(3) = normal;
                                      // pol(3) = mesh_data.Polygons(f, 3);
                                      neighPolygons.push_back(pol);
                                  }
                                  Eigen::Vector3<T> neighborVert = getPoint<T>(
                                      neighPolygons[0], neighPolygons[1], neighPolygons[2]);
                                  points.push_back(neighborVert);
                              }

                              Eigen::MatrixXd V1(points.size() * 3, 3);
                              Eigen::MatrixX<T> V2(points.size() * 3, 3);
                              int size = mesh_data.Hoods[v_idx].size();

                              for (int j = 0; j < size; j++) {
                                  int next = (j + 1) % size;
                                  int n1 = mesh_data.Hoods[v_idx][j];
                                  int n2 = mesh_data.Hoods[v_idx][next];
                                  Eigen::Vector3d originalNeighbor = data.V.row(n1);
                                  Eigen::Vector3<T> newNeighbor = points[j];
                                  Eigen::Vector3d originalNeighbor2 = data.V.row(n2);
                                  Eigen::Vector3<T> newNeighbor2 = points[next];
                                  V1.row(j * 3) = (ogVert - originalNeighbor);
                                  V2.row(j * 3) = (vert - newNeighbor);

                                  V1.row(j * 3 + 1) = (ogVert - originalNeighbor2);
                                  V2.row(j * 3 + 1) = (vert - newNeighbor2);

                                  V1.row(j * 3 + 2) = (originalNeighbor - originalNeighbor2);
                                  V2.row(j * 3 + 2) = (newNeighbor - newNeighbor2);
                              }

                              Eigen::Matrix3<T> Rot = getRotation<T>(V1, V2);
                              //wrong but will fix later
                              //TODO: fix later
                              T returnValue = 0;
                              for (int j = 0; j < size; j++) {
                                  int next = (j + 1) % size;
                                  int n1 = mesh_data.Hoods[v_idx][j];
                                  int n2 = mesh_data.Hoods[v_idx][next];
                                  Eigen::Vector3d ogNeighbor = data.V.row(n1);
                                  Eigen::Vector3d ogNeighbor2 = data.V.row(n2);
                                  Eigen::Vector3d v = ogNeighbor - ogVert;
                                  Eigen::Vector3<T> tv = points[j] - vert;

                                  Eigen::Vector3d v2 = ogNeighbor2 - ogVert;
                                  Eigen::Vector3<T> tv2 = points[next] - vert;

                                  Eigen::Vector3d v3 = ogNeighbor2 - ogNeighbor;
                                  Eigen::Vector3<T> tv3 = points[next] - points[j];

                                  returnValue += (tv - Rot * v).squaredNorm();
                                  returnValue += (tv2 - Rot * v2).squaredNorm();
                                  returnValue += (tv3 - Rot * v3).squaredNorm();
                              }
                              return returnValue;
                          });
    return func;
}

TinyAD::ScalarFunction<3, double, long> getBlockFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data) {
    auto func = TinyAD::scalar_function<3>(TinyAD::range(data.Polygons.rows()));

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
                                  for (auto v: mesh_data.Hoods[v_idx]) {
                                      for (auto p: mesh_data.VertPolygons[v]) {
                                          int size = faceSize(mesh_data.F.row(p));
                                          for (int fi = 0; fi < size; fi++) {
                                              if (mesh_data.F(p, fi) == data.b(i)) {
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
                                  for (auto k: mesh_data.VertPolygons[data.b(localConstrainsIndex[i])]) {
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
                                  for (auto k: mesh_data.VertPolygons[data.b(localConstrainsIndex[i])]) {
                                      Eigen::Vector4<T> pol;
                                      pol.head(3) = element.variables(k);
                                      pol(3) = dist(j);
                                      gons[k] = pol;
                                      j++;
                                  }
                              }

                              std::vector<Eigen::Vector4<T> > polygons;
                              for (auto f: mesh_data.VertPolygons[v_idx]) {
                                  Eigen::Vector4<T> pol;
                                  if (gons.find(f) != gons.end()) {
                                      pol = gons[f];
                                  } else {
                                      Eigen::Vector3<T> normal = element.variables(f);
                                      pol.head(3) = normal;
                                      pol(3) = mesh_data.Polygons(f, 3);
                                  }
                                  polygons.push_back(pol);
                              }
                              Eigen::Vector3<T> vert = getPoint<T>(polygons[0], polygons[1], polygons[2]);
                              Eigen::Vector3d ogVert = data.V.row(v_idx);
                              std::vector<Eigen::Vector3<T> > points;

                              for (auto neighbor: mesh_data.Hoods[v_idx]) {
                                  std::vector<Eigen::Vector4<T> > neighPolygons;
                                  for (auto f: mesh_data.VertPolygons[neighbor]) {
                                      Eigen::Vector4<T> pol;
                                      if (gons.find(f) != gons.end()) {
                                          pol = gons[f];
                                      } else {
                                          Eigen::Vector3<T> normal = element.variables(f);
                                          pol.head(3) = normal;
                                          pol(3) = mesh_data.Polygons(f, 3);
                                      }
                                      neighPolygons.push_back(pol);
                                  }
                                  Eigen::Vector3<T> neighborVert = getPoint<T>(
                                      neighPolygons[0], neighPolygons[1], neighPolygons[2]);
                                  points.push_back(neighborVert);
                              }

                              int size = mesh_data.Hoods[v_idx].size();

                              Eigen::Matrix3d Rot = data.R.block<3, 3>(0, v_idx * 3);

                              T returnValue = 0;
                              for (int j = 0; j < size; j++) {
                                  int next = (j + 1) % size;
                                  int n1 = mesh_data.Hoods[v_idx][j];
                                  int n2 = mesh_data.Hoods[v_idx][next];
                                  Eigen::Vector3d ogNeighbor = data.V.row(n1);
                                  Eigen::Vector3d ogNeighbor2 = data.V.row(n2);
                                  Eigen::Vector3d v = ogNeighbor - ogVert;
                                  Eigen::Vector3<T> tv = points[j] - vert;

                                  Eigen::Vector3d v2 = ogNeighbor2 - ogVert;
                                  Eigen::Vector3<T> tv2 = points[next] - vert;

                                  Eigen::Vector3d v3 = ogNeighbor2 - ogNeighbor;
                                  Eigen::Vector3<T> tv3 = points[next] - points[j];

                                  returnValue += (tv - Rot * v).squaredNorm();
                                  returnValue += (tv2 - Rot * v2).squaredNorm();
                                  returnValue += (tv3 - Rot * v3).squaredNorm();
                              }
                              return returnValue;
                          });
    return func;
}
