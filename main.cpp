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
#include <igl/stb/write_image.h>
#include <igl/stb/read_image.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
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

#define M_PI           3.14159265358979323846

typedef
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> >
RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
Eigen::MatrixXd U, Normals;
Eigen::MatrixXd Verts;
Eigen::MatrixXi Ff;
Eigen::MatrixXd V;
Eigen::VectorXi S, b;
Eigen::RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
igl::ARAPData arap_data;
custom_data custom_data;
poly_mesh_data mesh_data;

plane_arap_data plane_arap_data;
face_arap_data face_arap_data;

double point_x = 0;
double point_y = 0;
double point_z = 0;


void draw_face_mesh(igl::opengl::glfw::Viewer &viewer) {
    calculateTriangles(mesh_data);
    viewer.data().set_mesh(mesh_data.V, mesh_data.T);
    viewer.data().compute_normals();
}


void screenShot(igl::opengl::glfw::Viewer &viewer, std::string fileName) {
    //https://github.com/libigl/libigl/blob/f962e4a6b68afe978dc12a63702b7846a3e7a6ed/tutorial/607_ScreenCapture/main.cpp
    // Allocate temporary buffers
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(1280, 800);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(1280, 800);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(1280, 800);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(1280, 800);

    // Draw the scene in the buffers
    viewer.core().draw_buffer(
        viewer.data(), false, R, G, B, A);

    // Save it to a PNG
    igl::stb::write_image(fileName, R, G, B, A);
}


int main(int argc, char *argv[]) {
    std::ofstream measurementsFile;
    measurementsFile.open("../measurements/measurements.csv");
    using namespace Eigen;
    using namespace std;


    std::vector<std::vector<int> > polyF;

    std::string filePath;
    std::cout << "Enter the path to the PLY file: ";
    std::cin >> filePath;


    happly::PLYData plyIn(filePath);
    std::vector<std::array<double, 3> > vPos = plyIn.getVertexPositions();
    std::vector<std::vector<size_t> > fInd = plyIn.getFaceIndices<size_t>();
    V.conservativeResize(vPos.size(), 3);

    //convert happly datastructure
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

    // polyF = fInd;
    for (int i = 0; i < fInd.size(); i++) {
        std::vector<int> face;
        for (int j = 0; j < fInd[i].size(); j++) {
            face.push_back(fInd[i][j]);
        }
        polyF.push_back(face);
    }


    std::chrono::steady_clock::time_point meshPrecomputeTime = std::chrono::steady_clock::now();

    //precompute poly mesh: calculate planes and neighbors
    precompute_poly_mesh(mesh_data, V, polyF);

    std::chrono::steady_clock::time_point meshPrecomputeEndTime = std::chrono::steady_clock::now();
    long long meshPrecomputeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        meshPrecomputeEndTime - meshPrecomputeTime).count();
    std::cout << "mesh precompute: " << meshPrecomputeDuration << std::endl;

    igl::opengl::glfw::Viewer viewer;
    calcNewV(mesh_data);
    draw_face_mesh(viewer);


    Eigen::VectorXi con_idx(0);

    Eigen::VectorXi tempConP(1);
    Eigen::MatrixXd constraints(con_idx.size(), 3);
    Eigen::MatrixXd tempConstraints;
    for (int i = 0; i < con_idx.size(); i++) {
        constraints.row(i) = V.row(con_idx(i));
    }


    //settings
    std::atomic<bool> face(false);
    std::atomic<bool> conjugate(false);
    std::atomic<bool> useBlockFunc(false);
    std::atomic<bool> noInitalGuess(false);
    std::atomic<bool> triangleVersion(false);
    std::mutex m;
    bool newCons = false;
    bool changedCons = false;
    std::atomic<bool> onlyGradientDescent(false);
    std::atomic<bool> onlyDistantStep(false);


    //communication between threads
    std::atomic<bool> redraw(false);
    std::atomic<bool> measure(false);

    std::atomic<bool> screenshots(false);
    std::atomic<int> screenshotIt(-1);

    std::vector<int> screenshotIterations = {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};
    std::mutex m2;
    std::thread optimization_thread(
        [&]() {
            //ARAP precomputations

            b.conservativeResize(con_idx.size());
            for (int i = 0; i < con_idx.size(); i++) {
                b(i) = con_idx(i);
            }

            arap_data.energy = face.load(std::memory_order_relaxed)
                                   ? igl::ARAP_ENERGY_TYPE_ELEMENTS
                                   : igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
            face_arap_precomputation(mesh_data, face_arap_data, b);

            std::chrono::steady_clock::time_point arapPrecomputeTime = std::chrono::steady_clock::now();
            plane_arap_precomputation(mesh_data, plane_arap_data, b);
            std::chrono::steady_clock::time_point arapPrecomputeEndTime = std::chrono::steady_clock::now();
            long long arapPreComputationDuation = std::chrono::duration_cast<std::chrono::nanoseconds>(
                arapPrecomputeEndTime - arapPrecomputeTime).count();
            std::cout << "ARAP precomputation time: " << arapPreComputationDuation << std::endl;


            calculateTriangles(mesh_data);
            igl::arap_precomputation(mesh_data.V, mesh_data.T, 3, b, arap_data);


            auto funcBlock = getBlockFunction(constraints, mesh_data, plane_arap_data);
            auto faceFunc = getFaceFunction(constraints, mesh_data, face_arap_data);
            // auto func = getFunction(constraints, mesh_data, plane_arap_data);
            auto func = getEdgeFunction(constraints, mesh_data, plane_arap_data);


            Eigen::VectorXd x = func.x_from_data([&](int v_idx) {
                return mesh_data.Planes.row(v_idx);
            });
            Eigen::VectorXd x_block = funcBlock.x_from_data([&](int v_idx) {
                return mesh_data.Planes.row(v_idx).head(3);
            });
            // Projected Newton

            TinyAD::LinearSolver solver;
            int max_iters = 1000;
            double convergence_eps = 1e-2;
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > cg_solver;
            int i = -1;


            // measuring stuff

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point temp1 = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point temp2 = std::chrono::steady_clock::now();
            double tempEnergy = 0;

            while (true) {
                {
                    // no termination condition: easier to change constraints and settings without restarting the deformation
                    // in theory: terminate after energy change since last iteration is small enough
                    i++;
                    std::lock_guard<std::mutex> lock(m2);
                    bool switchedEnergy = (triangleVersion.load(std::memory_order_relaxed) && (
                                               arap_data.energy == igl::ARAP_ENERGY_TYPE_ELEMENTS) != face.load(
                                               std::memory_order_relaxed));
                    if (newCons || changedCons || switchedEnergy) {
                        //precompute again when constraints change or versions changed

                        begin = std::chrono::steady_clock::now();
                        i = 0;
                        if (newCons) {
                            con_idx.conservativeResize(tempConP.size());
                            for (int j = 0; j < tempConP.size(); j++) {
                                con_idx(j) = tempConP(j);
                            }
                            b.conservativeResize(con_idx.size());
                            for (int j = 0; j < con_idx.size(); j++) {
                                b(j) = con_idx(j);
                            }
                            // FACE
                            if (triangleVersion.load(std::memory_order_relaxed)) {
                                arap_data.energy = face.load(std::memory_order_relaxed)
                                                       ? igl::ARAP_ENERGY_TYPE_ELEMENTS
                                                       : igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
                                igl::arap_precomputation(mesh_data.originalV, mesh_data.T, 3, b, arap_data);
                            } else {
                                if (face.load(std::memory_order_relaxed)) {
                                    face_arap_precomputation(mesh_data, face_arap_data, b);
                                } else {
                                    plane_arap_precomputation(mesh_data, plane_arap_data, b);
                                }
                            }
                        } else if (switchedEnergy) {
                            arap_data.energy = face.load(std::memory_order_relaxed)
                                                   ? igl::ARAP_ENERGY_TYPE_ELEMENTS
                                                   : igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
                            igl::arap_precomputation(mesh_data.originalV, mesh_data.T, 3, b, arap_data);
                        }
                        constraints.conservativeResize(con_idx.size(), 3);
                        for (int j = 0; j < con_idx.size(); j++) {
                            constraints.row(j) = tempConstraints.row(j);
                        }
                        newCons = false;
                        changedCons = false;
                        if (screenshots.load(std::memory_order_relaxed)) {
                            calcNewV(mesh_data);
                            screenshotIt.store(-2, std::memory_order_relaxed);
                            redraw.store(true, std::memory_order_relaxed);
                            //wait for screenshot
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        }

                        if (onlyGradientDescent.load(std::memory_order_relaxed) && !triangleVersion.load(
                                std::memory_order_relaxed)) {
                            if (!noInitalGuess.load(std::memory_order_relaxed)) {
                                //initial guess
                                MatrixXd bc(b.size(), 3);
                                for (int j = 0; j < b.size(); j++) {
                                    bc.row(j) = mesh_data.V.row(b(j));
                                    for (int k = 0; k < con_idx.size(); k++) {
                                        if (con_idx(k) == b(j)) {
                                            bc.row(j) = constraints.row(k);
                                            break;
                                        }
                                    }
                                }
                                calcNewV(mesh_data);
                                //FACE
                                if (face.load(std::memory_order_relaxed)) {
                                    getFaceRotations(mesh_data, face_arap_data);
                                    global_face_distance_step(bc, mesh_data, face_arap_data);
                                } else {
                                    getRotations(mesh_data, plane_arap_data);
                                    global_distance_step(bc, mesh_data, plane_arap_data);
                                }
                            }
                        }
                    }
                }
                if (con_idx.size() <= 0) {
                    i = 0;
                    continue;
                }

                if (triangleVersion.load(std::memory_order_relaxed)) {
                    MatrixXd bc(b.size(), 3);
                    for (int j = 0; j < b.size(); j++) {
                        bc.row(j) = mesh_data.V.row(b(j));
                        for (int k = 0; k < con_idx.size(); k++) {
                            if (con_idx(k) == b(j)) {
                                bc.row(j) = constraints.row(k);
                                break;
                            }
                        }
                    }
                    igl::arap_solve(bc, arap_data, mesh_data.V);
                    redraw.store(true, std::memory_order_relaxed);
                    continue;
                }

                calcNewV(mesh_data);
                redraw.store(true, std::memory_order_relaxed);
                temp1 = std::chrono::steady_clock::now();


                double distanceStepImprovement = 0;
                if (!onlyGradientDescent.load(std::memory_order_relaxed)) {
                    //global distance step

                    tempEnergy = face.load(std::memory_order_relaxed)
                                     ? faceFunc.eval(x)
                                     : (useBlockFunc.load(std::memory_order_relaxed)
                                            ? funcBlock.eval(x_block)
                                            : func.eval(x));
                    MatrixXd bc(b.size(), 3);
                    for (int j = 0; j < b.size(); j++) {
                        bc.row(j) = mesh_data.V.row(b(j));
                        for (int k = 0; k < con_idx.size(); k++) {
                            if (con_idx(k) == b(j)) {
                                bc.row(j) = constraints.row(k);
                                break;
                            }
                        }
                    }
                    //FACE
                    if (face.load(std::memory_order_relaxed)) {
                        getFaceRotations(mesh_data, face_arap_data);
                        global_face_distance_step(bc, mesh_data, face_arap_data);
                    } else {
                        getRotations(mesh_data, plane_arap_data);
                        global_distance_step(bc, mesh_data, plane_arap_data);
                    }
                }


                calcNewV(mesh_data);
                temp2 = std::chrono::steady_clock::now();
                long long distanceTime = std::chrono::duration_cast<std::chrono::nanoseconds>(temp2 - temp1).count();
                bool doScreenshot = false;
                if (screenshots.load(std::memory_order_relaxed)) {
                    if (count(screenshotIterations.begin(), screenshotIterations.end(), i) > 0) {
                        doScreenshot = true;
                        screenshotIt.store(2 * i, std::memory_order_relaxed);
                        redraw.store(true, std::memory_order_relaxed);
                        //wait for screenshot
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    }
                }
                redraw.store(true, std::memory_order_relaxed);

                temp1 = std::chrono::steady_clock::now();

                if (face.load(std::memory_order_relaxed)) {
                    getFaceRotations(mesh_data, face_arap_data);
                } else {
                    getRotations(mesh_data, plane_arap_data);
                }


                // getRotations(mesh_data, plane_arap_data);
                x = func.x_from_data([&](int v_idx) {
                    return mesh_data.Planes.row(v_idx);
                });

                x_block = funcBlock.x_from_data([&](int v_idx) {
                    return mesh_data.Planes.row(v_idx).head(3).normalized();
                });


                if (onlyDistantStep.load(std::memory_order_relaxed)) {
                    double afterDistanceStepF = face.load(std::memory_order_relaxed)
                                                    ? faceFunc.eval(x)
                                                    : (useBlockFunc.load(std::memory_order_relaxed)
                                                           ? funcBlock.eval(x_block)
                                                           : func.eval(x));
                    TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << afterDistanceStepF);
                    continue;
                }

                VectorXd d;
                VectorXd g;
                double f;

                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                if (conjugate.load(std::memory_order_relaxed)) {
                    //truncated newton
                    Eigen::MatrixXd H_proj;
                    std::tie(f, g, H_proj) = face.load(std::memory_order_relaxed)
                                                 ? faceFunc.eval_with_hessian_proj(x)
                                                 : (useBlockFunc.load(std::memory_order_relaxed)
                                                        ? funcBlock.eval_with_hessian_proj(x_block)
                                                        : func.eval_with_hessian_proj(x));
                    d = cg_solver.compute(
                        H_proj + 1e-9 * TinyAD::identity<double>(
                            useBlockFunc.load(std::memory_order_relaxed) ? x_block.size() : x.size())).solve(-g);
                } else {
                    //gradient descent
                    std::tie(f, g) = face.load(std::memory_order_relaxed)
                                         ? faceFunc.eval_with_gradient(x)
                                         : (useBlockFunc.load(std::memory_order_relaxed)
                                                ? funcBlock.eval_with_gradient(x_block)
                                                : func.eval_with_gradient(x));
                    d = -g * 0.03;
                }

                TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
                double measureF = f;

                distanceStepImprovement = f - tempEnergy;
                tempEnergy = f;

                temp2 = std::chrono::steady_clock::now();
                long long gradientTime = std::chrono::duration_cast<std::chrono::nanoseconds>(temp2 - temp1).count();
                //measurement


                if (useBlockFunc.load(std::memory_order_relaxed)) {
                    x_block = TinyAD::line_search(x_block, d, f, g, funcBlock, 1.0, 0.5, 64);
                    f = funcBlock.eval(x_block);
                } else {
                    x = TinyAD::line_search(x, d, f, g, face.load(std::memory_order_relaxed) ? faceFunc : func, 1.0,
                                            0.5, 64);
                    f = face.load(std::memory_order_relaxed) ? faceFunc.eval(x) : func.eval(x);
                }

                double descentImprovement = f - tempEnergy;

                if (measure.load(std::memory_order_relaxed)) {
                    //write measurements
                    measurementsFile << i << " , "
                            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " , "
                            << distanceTime << " , "
                            << gradientTime << " , "
                            << measureF << " , "
                            << distanceStepImprovement << " , "
                            << descentImprovement;
                    measurementsFile << "\n";
                }


                if (useBlockFunc.load(std::memory_order_relaxed)) {
                    funcBlock.x_to_data(x_block, [&](int v_idx, const Eigen::VectorXd &p) {
                        mesh_data.Planes.row(v_idx).head(3) = p.normalized();
                    });
                } else {
                    func.x_to_data(x, [&](int v_idx, const Eigen::VectorXd &p) {
                        mesh_data.Planes.row(v_idx) = p;
                        mesh_data.Planes.row(v_idx).head(3) = mesh_data.Planes.row(v_idx).head(3).normalized();
                    });
                }

                // projection function
                for (int conIdx = 0; conIdx < con_idx.size(); conIdx++) {
                    Eigen::Vector4d vecs[3];
                    int j = 0;
                    for (auto k: mesh_data.FaceNeighbors[con_idx(conIdx)]) {
                        vecs[j] = mesh_data.Planes.row(k);
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
                    for (auto k: mesh_data.FaceNeighbors[con_idx(conIdx)]) {
                        mesh_data.Planes(k, 3) = dist(j);
                        j++;
                    }
                }

                if (doScreenshot) {
                    screenshotIt.store(2 * i + 1, std::memory_order_relaxed);
                    redraw.store(true, std::memory_order_relaxed);
                    //wait for screenshot
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
            TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));
        });


    // Plot the mesh with pseudocolors
    //draw_face_mesh(viewer, V);
    //viewer.data().set_mesh(U, F);
    viewer.core().is_animating = true;
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
                                     viewer.core().proj, viewer.core().viewport, mesh_data.V, mesh_data.T, fid, bc)) {
            int v_idx = mesh_data.T(fid, 0);
            float max = 0;
            for (int i = 0; i < 3; i++) {
                if (bc(i) > max) {
                    max = bc(i);
                    v_idx = mesh_data.T(fid, i);
                }
            }

            if (selectMode) {
                tempConP.conservativeResize(con_idx.size() + 1);
                for (int i = 0; i < con_idx.size(); i++) {
                    tempConP(i) = con_idx(i);
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
                point_x = mesh_data.V(v_idx, 0);
                point_y = mesh_data.V(v_idx, 1);
                point_z = mesh_data.V(v_idx, 2);
            } else {
                for (int i = 0; i < con_idx.size(); i++) {
                    if (con_idx(i) == v_idx) {
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
    //move constraints
    viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer &viewer, int, int) -> bool {
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        if (isDragging) {
            Eigen::Matrix4f modelView = viewer.core().view;

            Eigen::Vector4f startPointHom(startPoint(0), startPoint(1), startPoint(2), 1.0f);
            Eigen::Vector4f startPointView = modelView * startPointHom;
            double w = (viewer.core().proj * startPointView)(3);
            // magic numbers, too lazy to do it properly. But hey, it works
            startPointView(0) = startPointView(0) + (0.8 * w * (x - startX) / viewer.core().viewport(3));
            startPointView(1) = startPointView(1) + (0.8 * w * ((y - startY)) / viewer.core().viewport(3));
            Eigen::Matrix4f modelViewInv = modelView.inverse();
            Eigen::Vector4f backToWorld = modelViewInv * startPointView;
            tempConstraints.resize(con_idx.size(), 3);
            for (int i = 0; i < con_idx.size(); i++) {
                tempConstraints.row(i) = constraints.row(i);
            }
            Eigen::Vector3d newCon = ((backToWorld).head(3) / backToWorld(3)).cast<double>();
            tempConP = con_idx;
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
        return false;
    };

    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) -> bool {
        if (key == GLFW_KEY_SPACE) {
            selectMode = !selectMode;
        }
        if (key == '1') {
            screenShot(viewer, "out.png");
        }
        if (key == GLFW_KEY_S) {
            measurementsFile.close();
            measure.store(false, std::memory_order_relaxed);
        }
        if (key == GLFW_KEY_M) {
            //move last selected constraint (used for experiments)
            measure.store(true, std::memory_order_relaxed);
            tempConstraints.resize(con_idx.size(), 3);
            for (int i = 0; i < con_idx.size(); i++) {
                tempConstraints.row(i) = constraints.row(i);
            }
            Eigen::Vector3d newCon = tempConstraints.row(con_idx.size() - 1);
            newCon(0) = point_x;
            newCon(1) = point_y;
            newCon(2) = point_z;
            tempConP = con_idx;
            tempConstraints.row(con_idx.size() - 1) = newCon;
            viewer.data().clear_points();
            for (int i = 0; i < tempConstraints.rows(); i++) {
                if (i == con_idx.size() - 1) {
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
        if (redraw.load(std::memory_order_relaxed)) {
            if (triangleVersion.load(std::memory_order_relaxed)) {
                viewer.data().set_mesh(mesh_data.V, mesh_data.T);
                viewer.data().compute_normals();
            } else {
                draw_face_mesh(viewer);
            }

            //viewer.data().add_points(point, Eigen::RowVector3d(1, 0, 0));

            viewer.data().clear_points();
            for (int i = 0; i < constraints.rows(); i++) {
                viewer.data().add_points(constraints.row(i), Eigen::RowVector3d(1, 0, 0));
            }
            if (screenshotIt.load(std::memory_order_relaxed) >= 0) {
                int iteration = screenshotIt.load(std::memory_order_relaxed) / 2;
                std::string name = (screenshotIt.load(std::memory_order_relaxed) % 2 == 0)
                                       ? "distant_step" + std::to_string(iteration)
                                       : std::to_string(iteration);
                screenShot(viewer, name + ".png");
                screenshotIt.store(-1, std::memory_order_relaxed);
            } else if (screenshotIt.load(std::memory_order_relaxed) == -2) {
                screenShot(viewer, "handles.png");
            }
            redraw.store(false, std::memory_order_relaxed);
        }
        return false;
    };

    menu.callback_draw_custom_window = [&]() {
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiCond_FirstUseEver);
        ImGui::Begin(
            "New Window", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );

        ImGui::PushItemWidth(-80);
        ImGui::DragScalar("x", ImGuiDataType_Double, &point_x, 0.1, 0, 0, "%.4f");
        ImGui::DragScalar("y", ImGuiDataType_Double, &point_y, 0.1, 0, 0, "%.4f");
        ImGui::DragScalar("z", ImGuiDataType_Double, &point_z, 0.1, 0, 0, "%.4f");
        ImGui::PopItemWidth();

        bool conjugate_value = conjugate.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Conjugate", &conjugate_value)) {
            conjugate.store(conjugate_value, std::memory_order_relaxed);
        }

        bool face_value = face.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Face", &face_value)) {
            face.store(face_value, std::memory_order_relaxed);
        }

        bool block_value = useBlockFunc.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Block", &block_value)) {
            useBlockFunc.store(block_value, std::memory_order_relaxed);
        }

        bool gd_value = onlyGradientDescent.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Only Gradient Descent", &gd_value)) {
            onlyGradientDescent.store(gd_value, std::memory_order_relaxed);
        }

        bool ds_value = onlyDistantStep.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Only Distance Step", &ds_value)) {
            onlyDistantStep.store(ds_value, std::memory_order_relaxed);
        }

        bool initGuess_value = noInitalGuess.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("No initial guess", &initGuess_value)) {
            noInitalGuess.store(initGuess_value, std::memory_order_relaxed);
        }

        bool triangle_value = triangleVersion.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Use Triangles", &triangle_value)) {
            triangleVersion.store(triangle_value, std::memory_order_relaxed);
        }

        bool screenshot_value = screenshots.load(std::memory_order_relaxed);
        if (ImGui::Checkbox("Take Screenshots", &screenshot_value)) {
            screenshots.store(screenshot_value, std::memory_order_relaxed);
        }

        ImGui::End();
    };

    viewer.launch();
    if (optimization_thread.joinable()) {
        optimization_thread.join();
    }

    return 0;
}
