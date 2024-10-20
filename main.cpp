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
#include "face_arap.h"
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
face_arap_data face_arap_data;

//0.947214,   0.58541,         0,
//0.8090169943749473, -0.8090169943749473, 0.8090169943749473;
// double x = 0.8090169943749473;
// double y = -0.8090169943749473;
// double z = 0.8090169943749473;
//


void calcNewV(const Eigen::MatrixXd &poly) {
    for (int i = 0; i < mesh_data.V.rows(); i++) {
        std::vector<Eigen::Vector4d> pols;
        for (auto pol: mesh_data.VertPolygons[i]) {
            pols.push_back(poly.row(pol));
        }
        mesh_data.V.row(i) = getPoint<double>(pols[0], pols[1], pols[2]);
    }
}

void draw_face_mesh(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &poly) {
    calculateTriangles(mesh_data);
    viewer.data().set_mesh(mesh_data.V, mesh_data.T);
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


    Eigen::VectorXi conP(0);

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
    bool changedCons = false;
    bool onlyGradientDecent = false;
    std::mutex m2;
    std::thread optimization_thread(
        [&]() {
            Eigen::MatrixXd originalV = mesh_data.V;


            // U = centers;
            b.conservativeResize(conP.size());
            for (int i = 0; i < conP.size(); i++) {
                b(i) = conP(i);
            }

            face_arap_precomputation(mesh_data, face_arap_data, b);
            plane_arap_precomputation(mesh_data, plane_arap_data, b);


            // auto func = getFaceFunction(constraints, mesh_data, face_arap_data);
            auto func = getFunction(constraints, mesh_data, plane_arap_data);
            auto funcBlock = getBlockFunction(constraints, mesh_data, plane_arap_data);

            Eigen::VectorXd x = func.x_from_data([&](int v_idx) {
                // return Polygons.row(v_idx).head(3);
                return Polygons.row(v_idx);
            });
            Eigen::VectorXd x_block = funcBlock.x_from_data([&](int v_idx) {
                return Polygons.row(v_idx).head(3);
            });
            // Projected Newton

            TinyAD::LinearSolver solver;
            int max_iters = 1000;
            double convergence_eps = 1e-2;
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > cg_solver;
            int i = -1;
            bool useBlockFunc = false;
            while (true) {
                {
                    i++;
                    std::lock_guard<std::mutex> lock(m2);
                    if (newCons || changedCons) {
                        if (newCons) {
                            conP.conservativeResize(tempConP.size());
                            for (int j = 0; j < tempConP.size(); j++) {
                                conP(j) = tempConP(j);
                            }
                            b.conservativeResize(conP.size());
                            for (int j = 0; j < conP.size(); j++) {
                                b(j) = conP(j);
                            }
                            // mesh_data.V = face_arap_data.V;
                            // face_arap_precomputation(mesh_data, face_arap_data, b);
                            mesh_data.V = plane_arap_data.V;
                            plane_arap_precomputation(mesh_data, plane_arap_data, b);
                        }
                        constraints.conservativeResize(conP.size(), 3);
                        for (int j = 0; j < conP.size(); j++) {
                            constraints.row(j) = tempConstraints.row(j);
                        }
                        newCons = false;
                        changedCons = false;

                        if (onlyGradientDecent) {
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
                            calcNewV(mesh_data.Polygons);
                            // getFaceRotations(mesh_data, face_arap_data);
                            // global_face_distance_step(bc, mesh_data, face_arap_data);
                            getRotations(mesh_data, plane_arap_data);
                            global_distance_step(bc, mesh_data, plane_arap_data);
                        }
                    }
                }
                if (conP.size() <= 0) {
                    continue;
                }


                calcNewV(mesh_data.Polygons); {
                    std::lock_guard<std::mutex> lock(m);
                    redraw = true;
                }
                // std::this_thread::sleep_for(std::chrono::milliseconds(100));

                if (!onlyGradientDecent) {
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
                    // getFaceRotations(mesh_data, face_arap_data);
                    // global_face_distance_step(bc, mesh_data, face_arap_data);
                    getRotations(mesh_data, plane_arap_data);
                    global_distance_step(bc, mesh_data, plane_arap_data);
                }


                calcNewV(mesh_data.Polygons);
                Polygons = mesh_data.Polygons; {
                    std::lock_guard<std::mutex> lock(m);
                    redraw = true;
                }

                // getRotations(mesh_data, plane_arap_data);
                x = func.x_from_data([&](int v_idx) {
                    return mesh_data.Polygons.row(v_idx);
                });

                x_block = funcBlock.x_from_data([&](int v_idx) {
                    return mesh_data.Polygons.row(v_idx).head(3).normalized();
                });

                auto [f, g, H_proj] = useBlockFunc
                                          ? funcBlock.eval_with_hessian_proj(x_block)
                                          : func.eval_with_hessian_proj(x);
                TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
                Eigen::VectorXd d = cg_solver.compute(
                    H_proj + 1e-9 * TinyAD::identity<double>(useBlockFunc ? x_block.size() : x.size())).solve(-g);
                // Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
                if (TinyAD::newton_decrement(d, g) < convergence_eps) {
                    //break;
                }
                if (useBlockFunc) {
                    x_block = TinyAD::line_search(x_block, d, f, g, funcBlock);
                } else {
                    x = TinyAD::line_search(x, d, f, g, func);
                }


                if (useBlockFunc) {
                    funcBlock.x_to_data(x_block, [&](int v_idx, const Eigen::VectorXd &p) {
                        mesh_data.Polygons.row(v_idx).head(3) = p.normalized();
                    });
                } else {
                    func.x_to_data(x, [&](int v_idx, const Eigen::VectorXd &p) {
                        mesh_data.Polygons.row(v_idx) = p;
                        mesh_data.Polygons.row(v_idx).head(3) = mesh_data.Polygons.row(v_idx).head(3).normalized();
                    });
                }


                for (int conIdx = 0; conIdx < conP.size(); conIdx++) {
                    Eigen::Vector4d vecs[3];
                    int j = 0;
                    for (auto k: mesh_data.VertPolygons[conP(conIdx)]) {
                        vecs[j] = mesh_data.Polygons.row(k);
                        j++;
                    }
                    Eigen::Vector3d normal1 = vecs[0].head(3).normalized();
                    Eigen::Vector3d normal2 = vecs[1].head(3).normalized();
                    Eigen::Vector3d normal3 = vecs[2].head(3).normalized();

                    Eigen::Matrix3d normM;
                    normM.row(0) = normal1;
                    normM.row(1) = normal2;
                    normM.row(2) = normal3;

                    Eigen::Vector3d dist = normM * constraints.row(conIdx).transpose();
                    j = 0;
                    for (auto k: mesh_data.VertPolygons[conP(conIdx)]) {
                        mesh_data.Polygons(k, 3) = dist(j);
                        j++;
                    }
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
                                     viewer.core().proj, viewer.core().viewport, mesh_data.V, F, fid, bc)) {
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
                    tempConstraints.row(j) = mesh_data.V.row(tempConP(j));
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
                changedCons = true;
            }
        }
        return false; // Returning false lets the viewer process the event normally
    };

    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) -> bool {
        if (key == GLFW_KEY_SPACE) {
            selectMode = !selectMode;
        }
        if (key == GLFW_KEY_G) {
            onlyGradientDecent = !onlyGradientDecent;
        }
        if (key == GLFW_KEY_M) {
            tempConstraints.resize(conP.size(), 3);
            for (int i = 0; i < conP.size(); i++) {
                tempConstraints.row(i) = constraints.row(i);
            }
            Eigen::Vector3d newCon = tempConstraints.row(conP.size() - 1);
            newCon(1) = newCon(1) + 1;
            tempConP = conP;
            tempConstraints.row(conP.size() - 1) = newCon;
            viewer.data().clear_points();
            for (int i = 0; i < tempConstraints.rows(); i++) {
                if (i == conP.size() - 1) {
                    viewer.data().add_points(tempConstraints.row(i), Eigen::RowVector3d(0, 1, 0));
                    continue;
                }
                viewer.data().add_points(tempConstraints.row(i), Eigen::RowVector3d(1, 0, 0));
            } {
                std::lock_guard<std::mutex> lock(m2);
                changedCons = true;
            }
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
