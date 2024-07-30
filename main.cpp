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
//Eigen::MatrixXd V, U, Normals;
Eigen::MatrixXd Polygons;
Eigen::MatrixXd originalPolygons;
Eigen::MatrixXd rotations;
Eigen::MatrixXi F;
Eigen::MatrixXi Groups;
Eigen::MatrixXi VertsMap;
Eigen::MatrixXd Verts;
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


template<class T>
Eigen::Vector3<T> getPoint(Eigen::Vector4<T>& poly1, Eigen::Vector4<T>& poly2, Eigen::Vector4<T>& poly3) {
    Eigen::Vector3<T> normal1 = poly1.head(3).normalized();
    Eigen::Vector3<T> normal2 = poly2.head(3).normalized();
    Eigen::Vector3<T> normal3 = poly3.head(3).normalized();

    Eigen::Matrix3<T> m;
    m.row(0) = normal1;
    m.row(1) = normal2;
    m.row(2) = normal3;

    Eigen::Vector3<T> b;
    b << poly1(3), poly2(3), poly3(3);

    //Eigen::Vector3<T> x = m.colPivHouseholderQr().solve(b);
    Eigen::Vector3<T> x = m.partialPivLu().solve(b);
    //Eigen::Vector3<T> x = normal1 * poly1(3) + normal2 * poly2(3) + normal3 * poly3(3);
    return x;


}

void draw_face_mesh(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &poly) {
    using namespace Eigen;
    viewer.data().clear_points();
    int v = 0;
    Vf.resize(F.rows(), 3);
    // for (int i = 0; i < 6; i++) {
    //     viewer.data().add_points(points.row(i), Eigen::RowVector3d(1, 0, 0));
    // }
    for (int i = 0; i < F.rows(); i++) {
        auto row = F.row(i);
        Eigen::Vector4d p1 = poly.row(row(0));
        Eigen::Vector4d p2 = poly.row(row(1));
        Eigen::Vector4d p3 = poly.row(row(2));
        Vf.row(v) = getPoint<double>(p1,p2,p3);
        v++;
    }
    viewer.data().set_mesh(Vf, Ff);
}





int main(int argc, char *argv[]) {
    using namespace Eigen;
    using namespace std;
    //igl::readOFF("../cube.off", V, F);
    //U = V;
    //igl::readDMAT("../cube-selection.dmat", S);
    //igl::readDMAT("../cube-normals.dmat", Normals);
    igl::readDMAT("../cube-planes.dmat", Polygons);

    originalPolygons = Polygons;
    igl::readDMAT("../cube-connectivity.dmat", F);

    igl::readOFF("../plane-cube.off", Vf, Ff);
    //std::cout << Normals << std::endl;


    igl::opengl::glfw::Viewer viewer;
    draw_face_mesh(viewer, Polygons);
    //
    bool redraw = false;
    std::mutex m;
    std::thread optimization_thread(
        [&]() {

            Groups.resize(F.rows(), 6);
            VertsMap.resize(F.rows(), 4);
            Verts.resize(F.rows(),3);
            // Pre-compute stuff
            for(int i = 0; i < F.rows(); i++) {
                VertsMap(i,0) = i;

                Vector4d poly1 = Polygons.row(F(i,0));
                Vector4d poly2 = Polygons.row(F(i,1));
                Vector4d poly3 = Polygons.row(F(i,2));

                Verts.row(i) = getPoint(poly1,poly2,poly3);

                for(int j = 0; j< 3; j++) {

                    Groups(i,j)= F(i,j);
                    int a = j;
                    int b = (j+1) % 3;
                    for(int k = 0; k < F.rows(); k++) {
                        if(k == i) {
                            continue;
                        }
                        int c = 0;
                        int r = -1;
                        for(int v = 0; v < 3; v++) {
                            if(F(k,v) == F(i,a) || F(k,v) == F(i, b)) {
                                c++;
                            }
                            else {
                                r = F(k,v);
                            }

                        }
                        if(c>=2) {
                            Groups(i,j+3) = r;
                            VertsMap(i,j + 1) = k;
                        }
                    }

                }
            }


            // Set up function with 3D vertex positions as variables.
            auto func = TinyAD::scalar_function<4>(TinyAD::range(Polygons.rows()));

            // Add objective term per face. Each connecting 3 vertices.
            func.add_elements<6>(TinyAD::range(F.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                //calculate arap energy
                using T = TINYAD_SCALAR_TYPE(element);

                Eigen::Index f_idx = element.handle;
                Eigen::Vector4<T> vec0 = element.variables(Groups(f_idx,0));
                Eigen::Vector4<T> vec1 = element.variables(Groups(f_idx,1));
                Eigen::Vector4<T> vec2 = element.variables(Groups(f_idx,2));
                Eigen::Vector4<T> vec3 = element.variables(Groups(f_idx,3));
                Eigen::Vector4<T> vec4 = element.variables(Groups(f_idx,4));
                Eigen::Vector4<T> vec5 = element.variables(Groups(f_idx,5));

                //get points with current polygons
                Eigen::Vector3<T> point1 = getPoint<T>(vec0,vec1,vec2);
                Eigen::Vector3<T> point2 = getPoint(vec0,vec1,vec3);
                Eigen::Vector3<T> point3 = getPoint(vec1,vec2,vec4);
                Eigen::Vector3<T> point4 = getPoint(vec0,vec2,vec5);

                 Eigen::Vector3<T> a = point2 - point1;
                 Eigen::Vector3<T> b = point3 - point1;
                 Eigen::Vector3<T> c = point4 - point1;

                 //get points in original mesh
                 Eigen::Vector3d ogp1 = Verts.row(VertsMap(f_idx,0));
                 Eigen::Vector3d ogp2 = Verts.row(VertsMap(f_idx,1));
                 Eigen::Vector3d ogp3 = Verts.row(VertsMap(f_idx,2));
                 Eigen::Vector3d ogp4 = Verts.row(VertsMap(f_idx,3));

                 Eigen::Vector3d oa = ogp2 - ogp1;
                 Eigen::Vector3d ob = ogp3 - ogp1;
                 Eigen::Vector3d oc = ogp4 - ogp1;

                 Eigen::Matrix3d v1;
                 v1 << oa, ob, oc;

                 Eigen::Matrix3<T> v2;
                 v2 << a, b, c;

                 Eigen::Matrix3<T> S = v1 * v2.transpose();
                 Eigen::JacobiSVD<Eigen::Matrix3<T>> svd(S, Eigen::ComputeFullV | Eigen::ComputeFullU);
                 Eigen::Matrix3<T> U = svd.matrixU();
                 Eigen::Matrix3<T> E = svd.singularValues().asDiagonal();
                 Eigen::Matrix3<T> V = svd.matrixV();

                 Eigen::Matrix3<T> Rot = U * V.transpose();

                 Eigen::Vector3<T> ra = Rot * a;
                //T returnValue = pow((oa - ra).norm(),2);
                T returnValue = 0;

                // if(f_idx == 0) {
                //     Eigen::Vector3<T> targetPos;
                //     targetPos << 3, 6, 3;
                //     returnValue += pow(0.1 - (point1).norm(),2);
                // }
                // if(f_idx == 4) {
                // //     Eigen::Vector3<T> targetPos;
                // //     targetPos << -3, -3, -3;
                //     returnValue += pow(0.1 - (point1).norm(),2);
                // }

                return returnValue;


                //return pow(0.1 - vec0(3),2);
                //return pow(0.1 - (normal1 * vec0(3)).norm(),2);
                //return pow(0.1 - point1.norm(), 2);
                return (T)INFINITY;
            });

            // Assemble inital x vector from P matrix.
            // x_from_data(...) takes a lambda function that maps
            // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).

            Eigen::VectorXd x = func.x_from_data([&](int v_idx) {
                return Polygons.row(v_idx);
            });
            // Projected Newton

            TinyAD::LinearSolver solver;
            int max_iters = 100;
            double convergence_eps = 1e-2;
            for (int i = 0; i < max_iters; ++i) {
                auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
                TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
                Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
                if (TinyAD::newton_decrement(d, g) < convergence_eps)
                    break;
                x = TinyAD::line_search(x, d, f, g, func);

                func.x_to_data(x, [&](int v_idx, const Eigen::VectorXd &p) {
                    Polygons.row(v_idx) = p;
                    // V.row(v_idx) = p(seq(0, 2));
                    // Normals.row(v_idx) = p(seq(3, 5));
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


    viewer.callback_pre_draw = [&] (igl::opengl::glfw::Viewer& viewer)
    {
        if(redraw)
        {
            draw_face_mesh(viewer, Polygons);
          //viewer.data().set_vertices(P);
          //viewer.core().align_camera_center(P);
          viewer.core().camera_zoom = 2;
          {
            std::lock_guard<std::mutex> lock(m);
            redraw = false;
          }
        }
        return false;
    };
    // Customize the menu


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
    if(optimization_thread.joinable())
    {
        optimization_thread.join();
    }

    return 0;
}
