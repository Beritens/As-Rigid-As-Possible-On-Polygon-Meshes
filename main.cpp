#include <igl/colon.h>
#include <igl/directed_edge_orientations.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/PI.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/dqs.h>
#include <igl/readDMAT.h>
#include <igl/readOFF.h>
#include <igl/arap.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>
#include <mutex>

#include <TinyAD/ScalarFunction.hh>

#include "custom_arap.h"
#include "happly.h"
#include "plane_arap.h"
#include "poly_mesh_data.h"
#include "igl/unproject_onto_mesh.h"
#include "TinyAD/Utils/LinearSolver.hh"
#include "TinyAD/Utils/LineSearch.hh"
#include "TinyAD/Utils/NewtonDecrement.hh"
#include "TinyAD/Utils/NewtonDirection.hh"


typedef
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> >
RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
Eigen::MatrixXd U, Normals;
Eigen::MatrixXd Polygons;
Eigen::MatrixXd originalPolygons;
//Polygons around a vertex
//center of polygons
//group used for first method
//Neighbourhood of Polygon
//Neighbourhood Verts
Eigen::MatrixXd Verts;
Eigen::MatrixXd Vf;
Eigen::MatrixXi Ff;
Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::VectorXi S, b;
Eigen::RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
igl::ARAPData arap_data;
custom_data custom_data;
poly_mesh_data mesh_data;

plane_arap_data plane_arap_data;

//0.947214,   0.58541,         0,
//0.8090169943749473, -0.8090169943749473, 0.8090169943749473;
// double x = 0.8090169943749473;
// double y = -0.8090169943749473;
// double z = 0.8090169943749473;
//


namespace std {
    template<int N, typename T, bool B>
    struct numeric_limits<TinyAD::Scalar<N, T, B> > {
        static constexpr bool is_specialized = true;

        static TinyAD::Scalar<N, T, B> min() noexcept {
            return TinyAD::Scalar<N, T, B>{std::numeric_limits<T>::min()};
        }

        static TinyAD::Scalar<N, T, B> max() noexcept {
            return TinyAD::Scalar<N, T, B>{std::numeric_limits<T>::max()};
        }

        static TinyAD::Scalar<N, T, B> lowest() noexcept {
            return TinyAD::Scalar<N, T, B>{std::numeric_limits<T>::lowest()};
        }

        static constexpr int digits = std::numeric_limits<T>::digits;
        static constexpr int digits10 = std::numeric_limits<T>::digits10;
        static constexpr int max_digits10 = std::numeric_limits<T>::max_digits10;

        static constexpr bool is_signed = std::numeric_limits<T>::is_signed;
        static constexpr bool is_integer = std::numeric_limits<T>::is_integer;
        static constexpr bool is_exact = std::numeric_limits<T>::is_exact;

        static constexpr bool has_infinity = std::numeric_limits<T>::has_infinity;
        static constexpr bool has_quiet_NaN = std::numeric_limits<T>::has_quiet_NaN;
        static constexpr bool has_signaling_NaN = std::numeric_limits<T>::has_signaling_NaN;
        static constexpr float_denorm_style has_denorm = std::numeric_limits<T>::has_denorm;
        static constexpr bool has_denorm_loss = std::numeric_limits<T>::has_denorm_loss;

        static TinyAD::Scalar<N, T, B> infinity() noexcept {
            return TinyAD::Scalar<N, T, B>{std::numeric_limits<T>::infinity()};
        }

        static TinyAD::Scalar<N, T, B> quiet_NaN() noexcept {
            return TinyAD::Scalar<N, T, B>{std::numeric_limits<T>::quiet_NaN()};
        }

        static TinyAD::Scalar<N, T, B> signaling_NaN() noexcept {
            return TinyAD::Scalar<N, T, B>{std::numeric_limits<T>::signaling_NaN()};
        }

        static TinyAD::Scalar<N, T, B> denorm_min() noexcept {
            return TinyAD::Scalar<N, T, B>{std::numeric_limits<T>::denorm_min()};
        }

        static constexpr bool is_iec559 = std::numeric_limits<T>::is_iec559;
        static constexpr bool is_bounded = std::numeric_limits<T>::is_bounded;
        static constexpr bool is_modulo = std::numeric_limits<T>::is_modulo;

        static constexpr bool traps = std::numeric_limits<T>::traps;
        static constexpr bool tinyness_before = std::numeric_limits<T>::tinyness_before;
        static constexpr float_round_style round_style = std::numeric_limits<T>::round_style;
    };
}


void draw_face_mesh(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &poly) {
    using namespace Eigen;
    int v = 0;
    Vf.resize(mesh_data.V.rows(), 3);
    // for (int i = 0; i < 6; i++) {
    //     viewer.data().add_points(points.row(i), Eigen::RowVector3d(1, 0, 0));
    // }
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        // auto row = connections.row(i);
        std::vector<Vector4d> pols;
        for (auto pol: mesh_data.VertPolygons[i]) {
            pols.push_back(poly.row(pol));
        }
        Vf.row(i) = getPoint<double>(pols[0], pols[1], pols[2]);
        v++;
    }
    viewer.data().set_mesh(Vf, F);
}


void calculateFaceMatrixAndPolygons(Eigen::MatrixXi polyF, int face) {
    int size = faceSize(polyF.row(face));
    for (int j = 1; j < size - 1; j++) {
        F.conservativeResize(F.rows() + 1, 3);
        F(F.rows() - 1, 0) = polyF(face, 0);
        F(F.rows() - 1, 1) = polyF(face, j);
        F(F.rows() - 1, 2) = polyF(face, (j + 1) % size);
    }
    Polygons.conservativeResize(Polygons.rows() + 1, 4);
    Eigen::Vector3d pointa = V.row(polyF(face, 0));
    Eigen::Vector3d pointb = V.row(polyF(face, 1));
    Eigen::Vector3d pointc = V.row(polyF(face, 2));
    Eigen::Vector3d a = pointb - pointa;
    Eigen::Vector3d b = pointc - pointa;
    Eigen::Vector3d normal = a.cross(b).normalized();
    double dist = pointa.dot(normal);
    Eigen::Vector4d polygon;
    polygon << normal, dist;
    std::cout << face << std::endl;
    Polygons.row(Polygons.rows() - 1) = polygon;
}

// void calculateFaceCenters(Eigen::MatrixXi polyF, int face) {
//     centers.conservativeResize(centers.rows() + 1, 3);
//     Eigen::Vector3d center;
//     int size = faceSize(polyF.row(face));
//     for (int j = 0; j < size; j++) {
//         Eigen::Vector3d p = V.row(polyF(face, j));
//         center += p * (1.0 / (size * 1.0));
//     }
//     centers.row(face) = center;
// }
//
// void calculateHood(Eigen::MatrixXi polyF, int face) {
//     std::vector<int> neighbours;
//     Eigen::Vector3d center;
//     int size = faceSize(polyF.row(face));
//
//     //TODO: find faces that are also connected to last faces (makes finding connections easier)
//     int i = 0;
//     while (i < polyF.rows()) {
//         bool giveUp = i == face;
//         for (int j = 0; j < neighbours.size(); j++) {
//             if (i == neighbours[j]) {
//                 giveUp = true;
//             }
//         }
//         if (giveUp) {
//             i++;
//             continue;
//         }
//         int otherFaceSize = faceSize(polyF.row(i));
//         int lastFaceSize = -1;
//         if (neighbours.size() > 0) {
//             lastFaceSize = faceSize(polyF.row(neighbours[neighbours.size() - 1]));
//         }
//         int count = 0;
//         int count2 = 0;
//
//         for (int j = 0; j < otherFaceSize; j++) {
//             for (int k = 0; k < size; k++) {
//                 if (polyF(face, k) == polyF(i, j)) {
//                     count++;
//                     if (count >= 2) {
//                         break;
//                     }
//                 }
//             }
//             for (int k = 0; k < lastFaceSize; k++) {
//                 if (polyF(neighbours[neighbours.size() - 1], k) == polyF(i, j)) {
//                     count2++;
//                     if (count2 >= 2) {
//                         break;
//                     }
//                 }
//             }
//             if (count >= 2 && (count2 >= 2 || lastFaceSize < 0)) {
//                 break;
//             }
//         }
//         if (count >= 2 && (count2 >= 2 || lastFaceSize < 0)) {
//             neighbours.push_back(i);
//             i = 0;
//         } else {
//             i++;
//         }
//     }
//     Hoods[face] = neighbours;
// }
//
// void setConnectivityForOneVertex(Eigen::MatrixXi polyF, int v) {
//     connections.conservativeResize(connections.rows() + 1, 3);
//     int count = 0;
//     for (int i = 0; i < polyF.rows(); i++) {
//         if (count >= 3) {
//             return;
//         }
//         for (int j = 0; j < polyF.row(i).size(); j++) {
//             if (polyF(i, j) == v) {
//                 connections(connections.rows() - 1, count) = i;
//                 count++;
//                 break;
//             }
//         }
//     }
// }


void precomputeMesh(Eigen::MatrixXi polyF) {
    for (int i = 0; i < polyF.rows(); i++) {
        calculateFaceMatrixAndPolygons(polyF, i);
        // calculateFaceCenters(polyF, i);
        // calculateHood(polyF, i);
    }
    // for (int i = 0; i < V.rows(); i++) {
    //     setConnectivityForOneVertex(polyF, i);
    // }
    // //basically just faces, but making sure they are in the right order
    // for (int i = 0; i < Hoods.size(); i++) {
    //     std::vector<int> neighbours = Hoods[i];
    //     std::vector<int> verts;
    //     for (int j = 0; j < neighbours.size(); j++) {
    //         int k = (j + 1) % neighbours.size();
    //         std::vector<Eigen::VectorXi> faces;
    //         faces.push_back(polyF.row(i));
    //         faces.push_back(polyF.row(neighbours[j]));
    //         faces.push_back(polyF.row(neighbours[k]));
    //         std::set<int> shared_verts;
    //         std::set<int> shared_verts_temp;
    //
    //         for (int v: faces[0]) {
    //             if (v < 0) {
    //                 break;
    //             }
    //             shared_verts.insert(v);
    //         }
    //         for (int idx = 1; idx < 3; idx++) {
    //             for (int v: faces[idx]) {
    //                 if (v < 0) {
    //                     break;
    //                 }
    //                 if (shared_verts.count(v) > 0) {
    //                     shared_verts_temp.insert(v);
    //                 }
    //             }
    //             shared_verts = shared_verts_temp;
    //             shared_verts_temp = {};
    //         }
    //         verts.push_back(*shared_verts.begin());
    //     }
    //     HoodVerts[i] = verts;
    // }
    // std::cout<< centers << std::endl;
}

double getAngle(Eigen::Vector3d a, Eigen::Vector3d b) {
    return a.dot(b) / (a.norm() * b.norm());
}


int main(int argc, char *argv[]) {
    using namespace Eigen;
    using namespace std;


    Eigen::MatrixXi polyF;


    happly::PLYData plyIn("../test.ply");
    std::vector<std::array<double, 3> > vPos = plyIn.getVertexPositions();
    std::vector<std::vector<size_t> > fInd = plyIn.getFaceIndices<size_t>();
    V.conservativeResize(vPos.size(), 3);
    for (int i = 0; i < vPos.size(); i++) {
        V(i, 0) = vPos[i][0];
        V(i, 1) = vPos[i][1];
        V(i, 2) = vPos[i][2];
    }

    int Fcols = 0;
    for (int i = 0; i < fInd.size(); i++) {
        if (fInd[i].size() > Fcols) {
            Fcols = fInd[i].size();
        }
    }

    polyF.conservativeResize(fInd.size(), Fcols);
    for (int i = 0; i < fInd.size(); i++) {
        for (int j = 0; j < Fcols; j++) {
            if (fInd[i].size() > j) {
                polyF(i, j) = fInd[i][j];
            } else {
                polyF(i, j) = -1;
            }
        }
    }
    precomputeMesh(polyF);

    originalPolygons = Polygons;
    std::cout << Polygons << std::endl;
    //std::cout << Normals << std::endl;


    precompute_poly_mesh(mesh_data, V, polyF);
    igl::opengl::glfw::Viewer viewer;
    draw_face_mesh(viewer, Polygons);

    Eigen::MatrixXd V1_test(5, 3);
    Eigen::MatrixXd V2_test(5, 3);

    double theta = 10.0 * M_PI / 180.0;
    Eigen::Matrix3d rotationMatrix;
    rotationMatrix << std::cos(theta), -std::sin(theta), 0,
            std::sin(theta), std::cos(theta), 0,
            0, 0, 1;

    for (int i = 0; i < 5; i++) {
        V1_test.row(i) = mesh_data.V.row(i);
        V2_test.row(i) = rotationMatrix * mesh_data.V.row(i).transpose();
    }

    Matrix3d rotationTest = getRotation<double>(V1_test, V2_test);
    std::cout << rotationTest << std::endl;
    std::cout << rotationMatrix << std::endl;


    Eigen::VectorXi conP(1);
    // conP << 0, 12;
    conP << 0;

    Eigen::VectorXi tempConP(1);
    Eigen::MatrixXd lagrangeMultipliers = Eigen::MatrixXd::Zero(conP.size(), 4);
    Eigen::MatrixXd constraints(conP.size(), 3);
    Eigen::MatrixXd tempConstraints;
    for (int i = 0; i < conP.size(); i++) {
        constraints.row(i) = V.row(conP(i));
    }

    bool redraw = false;
    std::mutex m;
    bool newCons = false;
    std::mutex m2;
    std::thread optimization_thread(
        [&]() {
            // Groups.resize(connections.rows(), 6);
            // Eigen::MatrixXd cotanWeights;

            // VertsMap.resize(connections.rows(), 4);
            // Verts.resize(connections.rows(), 3);
            // Pre-compute stuff
            // cotanWeights.resize(V.rows(), V.rows());


            // U = centers;
            b.conservativeResize(conP.size());
            for (int i = 0; i < conP.size(); i++) {
                b(i) = conP(i);
            }
            //
            // igl::ARAPData arap_data;
            // arap_data.max_iter = 1;
            // if (!autodiff) {
            //     custom_arap_precomputation(centers, connections, centers.cols(), b, arap_data, custom_data, Polygons, V,
            //                                polyF);
            // }

            plane_arap_precomputation(mesh_data, plane_arap_data, b);

            auto func = TinyAD::scalar_function<4>(TinyAD::range(Polygons.rows()));

            func.add_elements<10>(TinyAD::range(mesh_data.V.rows()),
                                  [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                                      //calculate arap energy
                                      using T = TINYAD_SCALAR_TYPE(element);
                                      //TODO: einfach constraint berechnung hier rein packen, dann sollte es ja eig. gehen

                                      Eigen::Index v_idx = element.handle;
                                      std::vector<int> localConstrainsIndex;
                                      for (int i = 0; i < conP.size(); i++) {
                                          if (v_idx == conP(i)) {
                                              localConstrainsIndex.push_back(i);
                                          }
                                          for (auto v: mesh_data.Hoods[v_idx]) {
                                              for (auto p: mesh_data.VertPolygons[v]) {
                                                  int size = faceSize(mesh_data.F.row(p));
                                                  for (int fi = 0; fi < size; fi++) {
                                                      if (mesh_data.F(p, fi) == conP(i)) {
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
                                          for (auto k: mesh_data.VertPolygons[conP(localConstrainsIndex[i])]) {
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
                                                  m * constraints.row(localConstrainsIndex[i]).transpose();
                                          j = 0;
                                          for (auto k: mesh_data.VertPolygons[conP(localConstrainsIndex[i])]) {
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
                                      Eigen::Vector3d ogVert = mesh_data.V.row(v_idx);
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
                                      int i = 0;
                                      Eigen::MatrixXd V1(points.size(), 3);
                                      Eigen::MatrixX<T> V2(points.size(), 3);
                                      for (auto neighor: mesh_data.Hoods[v_idx]) {
                                          Eigen::Vector3d ogNeighbor = mesh_data.V.row(neighor);
                                          V1.row(i) = ogNeighbor - ogVert;
                                          V2.row(i) = points[i] - vert;

                                          i++;
                                      }
                                      Eigen::Matrix3<T> Rot = getRotation<T>(V1, V2);
                                      //wrong but will fix later
                                      //TODO: fix later
                                      T returnValue = 0;
                                      i = 0;
                                      for (auto neighor: mesh_data.Hoods[v_idx]) {
                                          Eigen::Vector3d ogNeighbor = mesh_data.V.row(neighor);
                                          Eigen::Vector3d v = ogNeighbor - ogVert;
                                          Eigen::Vector3<T> tv = points[i] - vert;

                                          returnValue += (tv - Rot * v).squaredNorm();

                                          i++;
                                      }
                                      return returnValue;
                                  });
            // }

            // Assemble inital x vector from P matrix.
            // x_from_data(...) takes a lambda function that maps
            // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).

            Eigen::VectorXd x = func.x_from_data([&](int v_idx) {
                // return Polygons.row(v_idx).head(3);
                return Polygons.row(v_idx);
            });
            // Projected Newton

            TinyAD::LinearSolver solver;
            int max_iters = 1000;
            double convergence_eps = 1e-2;
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > cg_solver;
            int i = -1;
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(m2);
                    if (newCons) {
                        conP.conservativeResize(tempConP.size());
                        for (int j = 0; j < tempConP.size(); j++) {
                            conP(j) = tempConP(j);
                        }
                        constraints.conservativeResize(conP.size(), 3);
                        for (int j = 0; j < conP.size(); j++) {
                            constraints.row(j) = tempConstraints.row(j);
                        }
                        b.conservativeResize(conP.size());
                        for (int j = 0; j < conP.size(); j++) {
                            b(j) = conP(j);
                        }
                        plane_arap_precomputation(mesh_data, plane_arap_data, b);
                        newCons = false;
                    }
                }
                i++;
                MatrixXd bc(b.size(), 3);
                for (int j = 0; j < b.size(); j++) {
                    bc.row(j) = mesh_data.V.row(b(j));
                    for (int k = 0; k < conP.size(); k++) {
                        if (conP(k) == b(j)) {
                            bc.row(j) = constraints.row(k);
                            break;
                        }
                    }
                }
                plane_arap_solve(bc, mesh_data, plane_arap_data);
                x = func.x_from_data([&](int v_idx) {
                    return mesh_data.Polygons.row(v_idx);
                });
                Polygons = mesh_data.Polygons;

                auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
                TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
                Eigen::VectorXd d = cg_solver.compute(H_proj + 1e-6 * TinyAD::identity<double>(x.size())).solve(-g);
                // Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
                if (TinyAD::newton_decrement(d, g) < convergence_eps) {
                    //break;
                }
                x = TinyAD::line_search(x, d, f, g, func);


                func.x_to_data(x, [&](int v_idx, const Eigen::VectorXd &p) {
                    mesh_data.Polygons.row(v_idx) = p;
                    mesh_data.Polygons.row(v_idx).head(3) = mesh_data.Polygons.row(v_idx).head(3).normalized();
                    // mesh_data.Polygons(v_idx, 0) = p(0);
                    // mesh_data.Polygons(v_idx, 1) = p(1);
                    // mesh_data.Polygons(v_idx, 2) = p(2);
                    //}
                    // V.row(v_idx) = p(seq(0, 2));
                    // Normals.row(v_idx) = p(seq(3, 5));
                    //P.row(v_idx) = p;
                });


                // for (int conIdx = 0; conIdx < conP.size(); conIdx++) {
                //     Eigen::Vector4d vecs[3];
                //     int j = 0;
                //     for (auto k: mesh_data.VertPolygons[conP(conIdx)]) {
                //         vecs[j] = mesh_data.Polygons.row(k);
                //         j++;
                //     }
                //     Eigen::Vector3d normal1 = vecs[0].head(3).normalized();
                //     Eigen::Vector3d normal2 = vecs[1].head(3).normalized();
                //     Eigen::Vector3d normal3 = vecs[2].head(3).normalized();
                //
                //     Eigen::Matrix3d normM;
                //     normM.row(0) = normal1;
                //     normM.row(1) = normal2;
                //     normM.row(2) = normal3;
                //
                //     Eigen::Vector3d dist = normM * constraints.row(conIdx).transpose();
                //     j = 0;
                //     for (auto k: mesh_data.VertPolygons[conP(conIdx)]) {
                //         mesh_data.Polygons(k, 3) = dist(j);
                //         j++;
                //     }
                // }
                // x = func.x_from_data([&](int v_idx) {
                //     return mesh_data.Polygons.row(v_idx);
                // });

                {
                    std::lock_guard<std::mutex> lock(m);
                    redraw = true;
                }
            }
            TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
            std::cout << Polygons << std::endl;

            // Write final x vector to P matrix.
            // x_to_data(...) takes a lambda function that writes the final value
            // of each variable (Eigen::Vector2d) back to our P matrix.
        });


    // Plot the mesh with pseudocolors
    //draw_face_mesh(viewer, V);
    //viewer.data().set_mesh(U, F);
    viewer.core().animation_max_fps = 60.;
    cout <<
            "Press [space] to toggle animation" << endl;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    bool selectMode = true;
    bool isDragging = false;
    int dragIndex = 0;
    double startX = 0;
    double startY = 0;
    Eigen::Vector3d startPoint;

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer &viewer, int, int) -> bool {
        int fid;
        Eigen::Vector3f bc;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view,
                                     viewer.core().proj, viewer.core().viewport, Vf, F, fid, bc)) {
            int v_idx = F(fid, 0);
            float max = 0;
            for (int i = 0; i < 3; i++) {
                if (bc(i) > max) {
                    max = bc(i);
                    v_idx = F(fid, i);
                }
            }

            if (selectMode) {
                tempConP.conservativeResize(conP.size() + 1);
                for (int i = 0; i < conP.size(); i++) {
                    tempConP(i) = conP(i);
                }

                tempConP(tempConP.size() - 1) = v_idx;
                tempConstraints.conservativeResize(tempConP.size(), 3);
                for (int j = 0; j < tempConP.size(); j++) {
                    tempConstraints.row(j) = Vf.row(tempConP(j));
                } {
                    std::lock_guard<std::mutex> lock(m2);
                    newCons = true;
                }
                viewer.data().clear_points();
                for (int i = 0; i < tempConstraints.rows(); i++) {
                    viewer.data().add_points(tempConstraints.row(i), Eigen::RowVector3d(1, 0, 0));
                }
            } else {
                for (int i = 0; i < conP.size(); i++) {
                    if (conP(i) == v_idx) {
                        isDragging = true;
                        dragIndex = i;
                        startPoint = constraints.row(i);
                        startX = x;
                        startY = y;
                        return true;
                    }
                }
            }
        }
        return false;
    };

    viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer &viewer, int button, int modifier) -> bool {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            isDragging = false;
        }
        return false;
    };
    viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer &viewer, int, int) -> bool {
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        if (isDragging) {
            Eigen::Matrix4f modelView = viewer.core().view;

            Eigen::Vector4f startPointHom(startPoint(0), startPoint(1), startPoint(2), 1.0f);
            Eigen::Vector4f startPointView = modelView * startPointHom;
            double w = (viewer.core().proj * startPointView)(3);
            startPointView(0) = startPointView(0) + (0.8 * w * (x - startX) / viewer.core().viewport(3));
            startPointView(1) = startPointView(1) + (0.8 * w * ((y - startY)) / viewer.core().viewport(3));
            std::cout << "viweport" << std::endl;
            std::cout << w << std::endl;
            std::cout << "end viewport" << std::endl;
            Eigen::Matrix4f modelViewInv = modelView.inverse();
            Eigen::Vector4f backToWorld = modelViewInv * startPointView;
            tempConstraints.resize(conP.size(), 3);
            for (int i = 0; i < conP.size(); i++) {
                tempConstraints.row(i) = constraints.row(i);
            }
            Eigen::Vector3d newCon = ((backToWorld).head(3) / backToWorld(3)).cast<double>();
            tempConP = conP;
            tempConstraints.row(dragIndex) = newCon;
            viewer.data().clear_points();
            for (int i = 0; i < tempConstraints.rows(); i++) {
                if (i == dragIndex) {
                    viewer.data().add_points(tempConstraints.row(i), Eigen::RowVector3d(0, 1, 0));
                    continue;
                }
                viewer.data().add_points(tempConstraints.row(i), Eigen::RowVector3d(1, 0, 0));
            } {
                std::lock_guard<std::mutex> lock(m2);
                newCons = true;
            }
        }
        return false; // Returning false lets the viewer process the event normally
    };

    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) -> bool {
        if (key == GLFW_KEY_SPACE) {
            selectMode = !selectMode;
        }
        return false;
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) {
        if (redraw) {
            draw_face_mesh(viewer, Polygons);

            //viewer.data().add_points(point, Eigen::RowVector3d(1, 0, 0));

            viewer.data().clear_points();
            for (int i = 0; i < constraints.rows(); i++) {
                viewer.data().add_points(constraints.row(i), Eigen::RowVector3d(1, 0, 0));
            }
            //viewer.data().set_vertices(P);
            //viewer.core().align_camera_center(P);
            //viewer.core().camera_zoom = 2;
            // {
            //     std::lock_guard<std::mutex> lock(m);
            //     constraints(1, 0) = x;
            //     constraints(1, 1) = y;
            //     constraints(1, 2) = z;
            redraw = false;
            // }
        }
        return false;
    };


    viewer.launch();
    if (optimization_thread.joinable()) {
        optimization_thread.join();
    }

    return 0;
}
