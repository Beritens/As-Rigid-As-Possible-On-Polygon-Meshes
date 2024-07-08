

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

#include "custom_arap.h"


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

void draw_face_mesh(igl::opengl::glfw::Viewer& viewer, const Eigen::MatrixXd& points) {

    viewer.data().clear_points();

    std::map<int, std::vector<int>> face_vert_map;
    int v=0;
    Vf.resize(F.rows(), 3);
    for (int i = 0; i < 6; i++) {
        viewer.data().add_points(points.row(i), Eigen::RowVector3d(1, 0, 0));
    }
    for (int i = 0; i < F.rows(); i++) {
        auto row = F.row(i);
        Eigen::Vector3d vert1 = rotations.block<3, 3>(0, row(0) * 3) * Normals.row(row(0)).transpose();
        Eigen::Vector3d vert2 = rotations.block<3, 3>(0, row(1) * 3) * Normals.row(row(1)).transpose();
        Eigen::Vector3d vert3 = rotations.block<3, 3>(0, row(2) * 3) * Normals.row(row(2)).transpose();
        //Eigen::Vector3d vert1 = Normals.row(row(0));
        //Eigen::Vector3d vert2 = Normals.row(row(1));
        //Eigen::Vector3d vert3 = Normals.row(row(2));

        for (int j = 0; j < 3; j++) {
            if (face_vert_map.find(row(j)) != face_vert_map.end()) {
                face_vert_map[row[j]].push_back(i);
            }
            else {
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

        b << points.row(row(0))* vert1, points.row(row(1)) * vert2, points.row(row(2)) * vert3;

        Eigen::Vector3d x = m.colPivHouseholderQr().solve(b);
        

        //viewer.data().add_points(x.transpose(), Eigen::RowVector3d(1, 0, 0)); 
        Vf.row(v) = x;
        v++;
    }
    //Ff.resize(12, 3);
    //int f = 0;
    //for (auto const& x : face_vert_map)
    //{
        //for (int i = 2; i < x.second.size(); i++) {
          //  Ff.row(f) << x.second[0] , x.second[i] , x.second[(i+1)];
            //f++;
        //}
        //Ff.row(f) << x.second[0], x.second[1], x.second[2];
        //f++;
        //Ff.row(f) << x.second[0], x.second[3], x.second[1];
        //f++;

    //}
    viewer.data().set_mesh(Vf, Ff);
}

bool pre_draw(igl::opengl::glfw::Viewer& viewer)
{
    using namespace Eigen;
    using namespace std;
    MatrixXd bc(b.size(), V.cols());
    for (int i = 0; i < b.size(); i++)
    {
        bc.row(i) = V.row(b(i));
        switch (S(b(i)))
        {
        case 0:
        {

            //const double r = 2 * 0.25;
            //bc(i, 0) += r * sin(0.2 * anim_t * 0.2 * igl::PI);
            //bc(i, 1) -= r + 0.2 * r * cos(igl::PI + 0.5 * anim_t * 2. * igl::PI);
            bc(i, 0) = x;
            bc(i, 1) = y;
            bc(i, 2) = z;
            break;
        }
        case 1:
        {
            //const double r = mid(1) * 0.15;
            //bc(i, 1) += r + r * cos(igl::PI + 0.15 * anim_t * 2. * igl::PI);
            //bc(i, 2) -= r * sin(0.15 * anim_t * 2. * igl::PI);
            break;
        }
        case 2:
        {
            const double r = mid(1) * 0.15;
            bc(i, 2) += r + r * cos(igl::PI + 0.35 * anim_t * 2. * igl::PI);
            bc(i, 0) += r * sin(0.35 * anim_t * 2. * igl::PI);
            break;
        }
        default:
            break;
        }
    }
    custom_arap_solve(bc, arap_data, U, rotations);
    std::cout << rotations.cols() << std::endl;
    draw_face_mesh(viewer, U);
    //viewer.data().set_vertices(U);
    //viewer.data().compute_normals();
    if (viewer.core().is_animating)
    {
        anim_t += anim_t_dir;
    }
    return false;
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mods)
{
    switch (key)
    {
    case ' ':
        viewer.core().is_animating = !viewer.core().is_animating;
        return true;
    }
    return false;
}

int main(int argc, char* argv[])
{
    using namespace Eigen;
    using namespace std;
    //igl::readOFF("C:/Users/brichter/Documents/CG/libigl-example-project/decimated-knight.off", V, F);
    igl::readOFF("C:/Users/brichter/Documents/CG/libigl-example-project/cube.off", V, F);
    U = V;
    //igl::readDMAT("C:/Users/brichter/Documents/CG/libigl-example-project/decimated-knight-selection.dmat", S);
    igl::readDMAT("C:/Users/brichter/Documents/CG/libigl-example-project/cube-selection.dmat", S);
    igl::readDMAT("C:/Users/brichter/Documents/CG/libigl-example-project/cube-normals.dmat", Normals);

    igl::readOFF("C:/Users/brichter/Documents/CG/libigl-example-project/plane-cube.off", Vf, Ff);
    std::cout << Normals << std::endl;

    // vertices in selection
    igl::colon<int>(0, V.rows() - 1, b);
    b.conservativeResize(stable_partition(b.data(), b.data() + b.size(),
        [](int i)->bool {return S(i) >= 0; }) - b.data());
    // Centroid
    mid = 0.5 * (V.colwise().maxCoeff() + V.colwise().minCoeff());
    // Precomputation
    arap_data.max_iter = 1;
    custom_arap_precomputation(V, F, V.cols(), b, arap_data);

    // Set color based on selection
    MatrixXd C(F.rows(), 3);
    RowVector3d purple(80.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0);
    RowVector3d gold(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0);
    for (int f = 0; f < F.rows(); f++)
    {
        if (S(F(f, 0)) >= 0 && S(F(f, 1)) >= 0 && S(F(f, 2)) >= 0)
        {
            C.row(f) = purple;
        }
        else
        {
            C.row(f) = gold;
        }
    }

    // Plot the mesh with pseudocolors
    igl::opengl::glfw::Viewer viewer;
    //draw_face_mesh(viewer, V);
    //viewer.data().set_mesh(U, F);
    viewer.data().set_colors(C);
    viewer.callback_pre_draw = &pre_draw;
    viewer.callback_key_down = &key_down;
    viewer.core().is_animating = false;
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
    menu.callback_draw_custom_window = [&]()
        {
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
