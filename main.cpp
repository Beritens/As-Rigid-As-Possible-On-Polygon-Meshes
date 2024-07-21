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
#include "TinyAD/Utils/LinearSolver.hh"
#include "TinyAD/Utils/LineSearch.hh"
#include "TinyAD/Utils/NewtonDecrement.hh"
#include "TinyAD/Utils/NewtonDirection.hh"


typedef
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> >
RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
Eigen::MatrixXd V, U, Normals;
Eigen::MatrixXd rotations;
Eigen::MatrixXi F;
Eigen::MatrixXd Vf;
Eigen::MatrixXi Ff;
Eigen::VectorXi S, b;
Eigen::RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
igl::ARAPData arap_data;

double x = 0;
double y = 0;
double z = 0;

void draw_face_mesh(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &points) {
    viewer.data().clear_points();

    std::map<int, std::vector<int> > face_vert_map;
    int v = 0;
    Vf.resize(F.rows(), 3);
    for (int i = 0; i < 6; i++) {
        viewer.data().add_points(points.row(i), Eigen::RowVector3d(1, 0, 0));
    }
    for (int i = 0; i < F.rows(); i++) {
        auto row = F.row(i);
        Eigen::Vector3d vert1 = rotations.block<3, 3>(0, row(0) * 3) * Normals.row(row(0)).transpose();
        Eigen::Vector3d vert2 = rotations.block<3, 3>(0, row(1) * 3) * Normals.row(row(1)).transpose();
        Eigen::Vector3d vert3 = rotations.block<3, 3>(0, row(2) * 3) * Normals.row(row(2)).transpose();

        for (int j = 0; j < 3; j++) {
            if (face_vert_map.find(row(j)) != face_vert_map.end()) {
                face_vert_map[row[j]].push_back(i);
            } else {
                std::vector<int> vec;
                vec.push_back(i);
                face_vert_map[row[j]] = vec;
            }
        }

        Eigen::Matrix3d m;
        m.row(0) = vert1;
        m.row(1) = vert2;
        m.row(2) = vert3;

        Eigen::Vector3d b;

        b << points.row(row(0)) * vert1, points.row(row(1)) * vert2, points.row(row(2)) * vert3;

        Eigen::Vector3d x = m.colPivHouseholderQr().solve(b);


        Vf.row(v) = x;
        v++;
    }
    viewer.data().set_mesh(Vf, Ff);
}

int main(int argc, char *argv[]) {
    using namespace Eigen;
    using namespace std;
    igl::readOFF("../cube.off", V, F);
    U = V;
    igl::readDMAT("../cube-selection.dmat", S);
    igl::readDMAT("../cube-normals.dmat", Normals);

    igl::readOFF("../plane-cube.off", Vf, Ff);
    std::cout << Normals << std::endl;


    //
    bool redraw = false;
    std::mutex m;
    std::thread optimization_thread(
        [&]() {
            // Pre-compute stuff


            // Set up function with 3D vertex positions as variables.
            auto func = TinyAD::scalar_function<6>(TinyAD::range(V.rows()));

            // Add objective term per face. Each connecting 3 vertices.
            func.add_elements<3>(TinyAD::range(F.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                //calculate arap energy
                using T = TINYAD_SCALAR_TYPE(element);

                Eigen::Index f_idx = element.handle;
                //
                // //get positions
                Eigen::Vector3<T> a = element.variables(F(f_idx, 0))(seq(0, 2));
                Eigen::Vector3<T> b = element.variables(F(f_idx, 1))(seq(0, 2));
                Eigen::Vector3<T> c = element.variables(F(f_idx, 2))(seq(0, 2));
                //
                //get normals
                Eigen::Vector3<T> an = element.variables(F(f_idx, 0))(seq(3, 5));
                Eigen::Vector3<T> bn = element.variables(F(f_idx, 1))(seq(3, 5));
                Eigen::Vector3<T> cn = element.variables(F(f_idx, 2))(seq(3, 5));
                //
                Eigen::Matrix3<T> mat;
                mat.row(0) = an;
                mat.row(1) = bn;
                mat.row(2) = cn;
                //
                Eigen::Vector3<T> r;
                // //
                // r << a * an, b * bn, c * cn;
                // //
                // Eigen::Vector3<T> _x = m.colPivHouseholderQr().solve(r);
                //
                // //just a small test
                return a[0];
                //return (T)INFINITY;
            });

            // Assemble inital x vector from P matrix.
            // x_from_data(...) takes a lambda function that maps
            // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).

            Eigen::VectorXd x = func.x_from_data([&](int v_idx) {
                VectorXd vec_joined(6);
                vec_joined << V.row(v_idx).transpose(), Normals.row(v_idx).transpose();
                return vec_joined;
            });
            // Projected Newton
            TinyAD::LinearSolver solver;
            int max_iters = 1000;
            double convergence_eps = 1e-2;
            for (int i = 0; i < max_iters; ++i) {
                auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
                TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
                Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
                if (TinyAD::newton_decrement(d, g) < convergence_eps)
                    break;
                x = TinyAD::line_search(x, d, f, g, func);
                func.x_to_data(x, [&](int v_idx, const Eigen::VectorXd &p) {
                    V.row(v_idx) = p(seq(0, 2));
                    Normals.row(v_idx) = p(seq(3, 5));
                    //P.row(v_idx) = p;
                }); {
                    std::lock_guard<std::mutex> lock(m);
                    redraw = true;
                }
            }
            TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));

            // Write final x vector to P matrix.
            // x_to_data(...) takes a lambda function that writes the final value
            // of each variable (Eigen::Vector2d) back to our P matrix.
        });


    // Plot the mesh with pseudocolors
    igl::opengl::glfw::Viewer viewer;
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


    // Customize the menu
    double doubleVariable = 0.1f; // Shared between two menus


    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiCond_FirstUseEver);
        ImGui::Begin(
            "New Window", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );

        // Expose the same variable directly ...
        ImGui::PushItemWidth(-80);
        ImGui::DragScalar("x", ImGuiDataType_Double, &x, 0.1, 0, 0, "%.4f");
        ImGui::DragScalar("y", ImGuiDataType_Double, &y, 0.1, 0, 0, "%.4f");
        ImGui::DragScalar("z", ImGuiDataType_Double, &z, 0.1, 0, 0, "%.4f");
        ImGui::PopItemWidth();


        ImGui::End();
    };

    viewer.launch();
}
