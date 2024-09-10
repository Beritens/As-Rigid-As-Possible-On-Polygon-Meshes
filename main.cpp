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
Eigen::MatrixXd rotations;
Eigen::MatrixXi connections;
Eigen::MatrixXd centers;
Eigen::MatrixXi Groups;
Eigen::MatrixXi VertsMap;
Eigen::MatrixXi PolyMap;
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

    //0.947214,   0.58541,         0,
    //0.8090169943749473, -0.8090169943749473, 0.8090169943749473;
// double x = 0.8090169943749473;
// double y = -0.8090169943749473;
// double z = 0.8090169943749473;
//
double x = 0;
double y =  0;
double z = 0;
bool autodiff = false;

namespace std {
    template <>
    struct numeric_limits<TinyAD::Scalar<24, double, true>> {
        static constexpr bool is_specialized = true;

        static TinyAD::Scalar<24, double, true> min() noexcept {
            return TinyAD::Scalar<24, double, true>{std::numeric_limits<double>::min()};
        }

        static TinyAD::Scalar<24, double, true> max() noexcept {
            return TinyAD::Scalar<24, double, true>{std::numeric_limits<double>::max()};
        }

        static TinyAD::Scalar<24, double, true> lowest() noexcept {
            return TinyAD::Scalar<24, double, true>{std::numeric_limits<double>::lowest()};
        }

        static constexpr int digits = std::numeric_limits<double>::digits;
        static constexpr int digits10 = std::numeric_limits<double>::digits10;
        static constexpr int max_digits10 = std::numeric_limits<double>::max_digits10;

        static constexpr bool is_signed = std::numeric_limits<double>::is_signed;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;

        static constexpr bool has_infinity = std::numeric_limits<double>::has_infinity;
        static constexpr bool has_quiet_NaN = std::numeric_limits<double>::has_quiet_NaN;
        static constexpr bool has_signaling_NaN = std::numeric_limits<double>::has_signaling_NaN;
        static constexpr float_denorm_style has_denorm = std::numeric_limits<double>::has_denorm;
        static constexpr bool has_denorm_loss = std::numeric_limits<double>::has_denorm_loss;

        static TinyAD::Scalar<24, double, true> infinity() noexcept {
            return TinyAD::Scalar<24, double, true>{std::numeric_limits<double>::infinity()};
        }

        static TinyAD::Scalar<24, double, true> quiet_NaN() noexcept {
            return TinyAD::Scalar<24, double, true>{std::numeric_limits<double>::quiet_NaN()};
        }

        static TinyAD::Scalar<24, double, true> signaling_NaN() noexcept {
            return TinyAD::Scalar<24, double, true>{std::numeric_limits<double>::signaling_NaN()};
        }

        static TinyAD::Scalar<24, double, true> denorm_min() noexcept {
            return TinyAD::Scalar<24, double, true>{std::numeric_limits<double>::denorm_min()};
        }

        static constexpr bool is_iec559 = std::numeric_limits<double>::is_iec559;
        static constexpr bool is_bounded = std::numeric_limits<double>::is_bounded;
        static constexpr bool is_modulo = std::numeric_limits<double>::is_modulo;

        static constexpr bool traps = std::numeric_limits<double>::traps;
        static constexpr bool tinyness_before = std::numeric_limits<double>::tinyness_before;
        static constexpr float_round_style round_style = std::numeric_limits<double>::round_style;
    };
}



template<class T>
Eigen::Vector3<T> getPoint(Eigen::Vector4<T> &poly1, Eigen::Vector4<T> &poly2, Eigen::Vector4<T> &poly3) {
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


template<class T>
Eigen::Matrix3<T> getRotation(Eigen::MatrixX<T> v1, Eigen::MatrixX<T> v2) {
    // std::cout << v1 << std::endl;
    Eigen::Matrix3<T> S = v1.transpose() * v2;
    Eigen::JacobiSVD<Eigen::Matrix3<T> > svd(S, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3<T> U = svd.matrixU();
    Eigen::Matrix3<T> V = svd.matrixV();
    //
    //
    Eigen::Matrix3<T> Rot = U * V.transpose();
    // Eigen::Matrix3<T> Rot;
    // if (Rot.determinant() < 0) {
    //     Eigen::Matrix3<T> I = Eigen::Matrix3<T>::Identity();
    //     I(2, 2) = -1;
    //     Rot = U * I * V.transpose();
    // }
    return Rot;
}


void draw_face_mesh(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd &poly) {
    using namespace Eigen;
    viewer.data().clear_points();
    int v = 0;
    Vf.resize(connections.rows(), 3);
    // for (int i = 0; i < 6; i++) {
    //     viewer.data().add_points(points.row(i), Eigen::RowVector3d(1, 0, 0));
    // }
    for (int i = 0; i < connections.rows(); i++) {
        auto row = connections.row(i);
        Eigen::Vector4d p1 = poly.row(row(0));
        Eigen::Vector4d p2 = poly.row(row(1));
        Eigen::Vector4d p3 = poly.row(row(2));
        Vf.row(v) = getPoint<double>(p1, p2, p3);
        v++;
    }
    viewer.data().set_mesh(Vf, F);
}

int faceSize(Eigen::VectorXi face) { int count = 0;
    for (int i = 0; i < face.size(); ++i) {
        if (face(i) != -1) {
            count++;
        }
    }
    return count;
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

void calculateFaceCenters(Eigen::MatrixXi polyF, int face) {
    centers.conservativeResize(centers.rows() + 1, 3);
    Eigen::Vector3d center;
    int size = faceSize(polyF.row(face));
    for (int j = 0; j < size; j++) {
        Eigen::Vector3d p = V.row(polyF(face,j));
        center += p * (1.0 / (size * 1.0));
    }
    centers.row(face) = center;
}

void setConnectivityForOneVertex(Eigen::MatrixXi polyF, int v) {
    connections.conservativeResize(connections.rows() + 1, 3);
    int count = 0;
    for (int i = 0; i < polyF.rows(); i++) {
        if (count >= 3) {
            return;
        }
        for (int j = 0; j < polyF.row(i).size(); j++) {
            if (polyF(i, j) == v) {
                connections(connections.rows() - 1, count) = i;
                count++;
                break;
            }
        }
    }
}


void precomputeMesh(Eigen::MatrixXi polyF) {
    for (int i = 0; i < polyF.rows(); i++) {
        calculateFaceMatrixAndPolygons(polyF, i);
        calculateFaceCenters(polyF, i);
    }
    for (int i = 0; i < V.rows(); i++) {
        setConnectivityForOneVertex(polyF, i);
    }
    // std::cout<< centers << std::endl;
}

double getAngle(Eigen::Vector3d a, Eigen::Vector3d b) {
    return a.dot(b) / (a.norm() * b.norm());
}



int main(int argc, char *argv[]) {
    using namespace Eigen;
    using namespace std;
    //igl::readOFF("../cube.off", V, F);
    //U = V;
    //igl::readDMAT("../cube-selection.dmat", S);
    //igl::readDMAT("../cube-normals.dmat", Normals);
    //igl::readDMAT("../cube-planes.dmat", Polygons);

    //igl::readDMAT("../cube-connectivity.dmat", connections);

    //igl::readOFF("../plane-cube.off", Vf, Ff);
    Eigen::MatrixXi polyF;

    //igl::readOFF("../Dodecahedron.off", V, polyF);

    happly::PLYData plyIn("../complex.ply");
    std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
    std::vector<std::vector<size_t>> fInd = plyIn.getFaceIndices<size_t>();
    V.conservativeResize(vPos.size(),3);
    for(int i = 0; i < vPos.size(); i++) {
        V(i,0) = vPos[i][0];
        V(i,1) = vPos[i][1];
        V(i,2) = vPos[i][2];
    }

    int Fcols = 0;
    for(int i = 0; i < fInd.size(); i++) {
        if(fInd[i].size() > Fcols) {
            Fcols = fInd[i].size();
        }
    }

    polyF.conservativeResize(fInd.size(),Fcols);
    for(int i = 0; i < fInd.size(); i++) {
        for(int j = 0; j< Fcols; j++) {
            if(fInd[i].size() > j) {
                polyF(i,j) = fInd[i][j];
            } else {
                polyF(i,j) = -1;
            }
        }
    }
    precomputeMesh(polyF);

    originalPolygons = Polygons;
    std::cout << Polygons << std::endl;
    //std::cout << Normals << std::endl;




    igl::opengl::glfw::Viewer viewer;
    draw_face_mesh(viewer, Polygons);

    Matrix3d rotationTest = getRotation<double>(V, V);
    std::cout << rotationTest << std::endl;


    Eigen::VectorXi conP(2); conP << 0, 22;
    Eigen::MatrixXd lagrangeMultipliers = Eigen::MatrixXd::Zero(conP.size(),4);
    Eigen::MatrixXd constraints(conP.size(),3);
    for(int i = 0; i < conP.size(); i++) {
        if(autodiff) {
            constraints.row(i) = V.row(conP(i));

        }
        else {
            constraints.row(i) = centers.row(conP(i));
        }
    }
    x = constraints(1,0);
    y = constraints(1,1);
    z = constraints(1,2);

    bool redraw = false;
    std::mutex m;
    std::thread optimization_thread(
        [&]() {
            Groups.resize(connections.rows(), 6);
            PolyMap.resize(V.rows(), 3);
            Eigen::MatrixXd cotanWeights;

            VertsMap.resize(connections.rows(), 4);
            Verts.resize(connections.rows(), 3);
            // Pre-compute stuff
            cotanWeights.resize(V.rows(), V.rows());


            U = centers;
            b.conservativeResize(conP.size());
            for(int i = 0; i<conP.size(); i++) {
                b(i) = conP(i);
            }

            igl::ARAPData arap_data;
            arap_data.max_iter = 1;
            custom_arap_precomputation(centers,connections, centers.cols(), b, arap_data, custom_data, Polygons);


            for (int i = 0; i < connections.rows(); i++) {
                VertsMap(i, 0) = i;

                Vector4d poly1 = Polygons.row(connections(i, 0));
                Vector4d poly2 = Polygons.row(connections(i, 1));
                Vector4d poly3 = Polygons.row(connections(i, 2));

                Verts.row(i) = getPoint(poly1, poly2, poly3);
                Vector3i polies; polies << connections(i,0), connections(i,1), connections(1,2);
                PolyMap.row(i) = polies;

                for (int j = 0; j < 3; j++) {
                    Groups(i, j) = connections(i, j);
                    int a = j;
                    int b = (j + 1) % 3;
                    for (int k = 0; k < connections.rows(); k++) {
                        if (k == i) {
                            continue;
                        }
                        int c = 0;
                        int r = -1;
                        for (int v = 0; v < 3; v++) {
                            if (connections(k, v) == connections(i, a) || connections(k, v) == connections(i, b)) {
                                c++;
                            } else {
                                r = connections(k, v);
                            }
                        }
                        if (c >= 2) {
                            Groups(i, j + 3) = r;
                            VertsMap(i, j + 1) = k;
                        }
                    }
                }

                for (int j = 0; j < 3; j++) {
                    //calculate cotan weight

                    Vector3d origin = V.row(VertsMap(i, 0));

                    Vector3d a = V.row(VertsMap(i, (j + 0) % 3 + 1));
                    Vector3d b = V.row(VertsMap(i, (j + 1) % 3 + 1));
                    Vector3d c = V.row(VertsMap(i, (j + 2) % 3 + 1));

                    double angle1 = getAngle(origin - b, a - b);
                    double angle2 = getAngle(origin - c, a - c);

                    double weight = 1.0 / tan(angle1) + 1.0 / tan(angle2);

                    cotanWeights(i, VertsMap(i, j + 1)) = weight;
                }
            }

            //test stuff
            // Eigen::Matrix3d scaleMatrix;
            // scaleMatrix << 1.5, 0, 0,
            //         0, 3, 0,
            //         0, 0, 0.5;
            //
            // Eigen::Matrix3d isoScaleMatrix;
            // isoScaleMatrix << 3.4, 0, 0,
            //         0, 1.4, 0,
            //         0, 0, 2.4;

            // Set up function with 3D vertex positions as variables.
            auto func = TinyAD::scalar_function<4>(TinyAD::range(Polygons.rows() + conP.size()));

            func.add_elements<4>(TinyAD::range(conP.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {

                using T = TINYAD_SCALAR_TYPE(element);
                Eigen::Index f_idx = element.handle;
                Eigen::Vector4<T> lagrangeM = element.variables(f_idx + Polygons.rows());
                Eigen::Vector4<T> vecs[3];
                for(int i = 0; i<3; i++) {
                    vecs[i] = element.variables(Groups(conP(f_idx), i));
                }
                Eigen::Vector3<T> point = getPoint<T>(vecs[0], vecs[1], vecs[2]);
                Eigen::Vector3d target = constraints.row(f_idx);
                return lagrangeM(0) * (target - point).squaredNorm();

            });

            // Add objective term per face. Each connecting 3 vertices.
            func.add_elements<6>(TinyAD::range(connections.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                //calculate arap energy
                using T = TINYAD_SCALAR_TYPE(element);

                Eigen::Index f_idx = element.handle;

                Eigen::Vector4<T> vecs[6];
                for(int i = 0; i<6; i++) {

                    vecs[i] = element.variables(Groups(f_idx, i));
                }

                //get points with current polygons
                Eigen::Vector3<T> point1 = getPoint<T>(vecs[0], vecs[1], vecs[2]);
                Eigen::Vector3<T> point2 = getPoint<T>(vecs[0], vecs[1], vecs[3]);
                Eigen::Vector3<T> point3 = getPoint<T>(vecs[1], vecs[2], vecs[4]);
                Eigen::Vector3<T> point4 = getPoint<T>(vecs[0], vecs[2], vecs[5]);

                Eigen::Vector3<T> a = point2 - point1;
                Eigen::Vector3<T> b = point3 - point1;
                Eigen::Vector3<T> c = point4 - point1;

                //get points in original mesh
                Eigen::Vector3d ogp1 = Verts.row(VertsMap(f_idx, 0));
                Eigen::Vector3d ogp2 = Verts.row(VertsMap(f_idx, 1));
                Eigen::Vector3d ogp3 = Verts.row(VertsMap(f_idx, 2));
                Eigen::Vector3d ogp4 = Verts.row(VertsMap(f_idx, 3));

                Eigen::Vector3d oa = ogp2 - ogp1;
                Eigen::Vector3d ob = ogp3 - ogp1;
                Eigen::Vector3d oc = ogp4 - ogp1;

                Eigen::Matrix3d v1;
                v1 << oa, ob, oc;

                Eigen::Matrix3<T> v2;
                v2 << a, b, c;

                Eigen::Matrix3<double> Rot2 = getRotation<double>(v1, v1);
                //RealSvd2x2 caused problem (d was 0 but wasn't catched?) if (abs(d) < (std::numeric_limits<RealScalar>::min)() || d == 0) {
                Eigen::Matrix3<T> Rot = getRotation<T>(v1, v2);

                //return 0;


                //return ((scaleMatrix * ogp1) - point1).squaredNorm();

                T returnValue = 0;
                //return returnValue;

                // if (f_idx == 5 || f_idx == 2) {
                //     returnValue = 100 * ((isoScaleMatrix * ogp1) - point1).squaredNorm();
                //     //return returnValue;
                // }

                // if(f_idx == 12) {
                //
                //     Eigen::Vector3d target;
                //     target << x,y,z;
                //     returnValue = 1000 * ((target) - point1).squaredNorm();
                // }
                //return ((ogp1) - point1).squaredNorm();
                //
                // //return (T) INFINITY;
                //
                //
                Eigen::Vector3<T> ra = Rot * a;
                Eigen::Vector3<T> rb = Rot * b;
                Eigen::Vector3<T> rc = Rot * c;
                returnValue += cotanWeights(VertsMap(f_idx, 0), VertsMap(f_idx, 1)) * (oa - ra).squaredNorm();
                returnValue += cotanWeights(VertsMap(f_idx, 0), VertsMap(f_idx, 2)) * (ob - rb).squaredNorm();
                returnValue += cotanWeights(VertsMap(f_idx, 0), VertsMap(f_idx, 3)) * (oc - rc).squaredNorm();
                return returnValue;
                // T returnValue = 0;
            });

            // Assemble inital x vector from P matrix.
            // x_from_data(...) takes a lambda function that maps
            // each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).

            Eigen::VectorXd x = func.x_from_data([&](int v_idx) {
                if(v_idx >= Polygons.rows()) {
                    return lagrangeMultipliers.row(v_idx - Polygons.rows());
                }
                return Polygons.row(v_idx);
            });
            // Projected Newton

            TinyAD::LinearSolver solver;
            int max_iters = 1000;
            double convergence_eps = 1e-2;
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > cg_solver;
            int i = -1;
            while (true) {
                i++;
                if(autodiff) {
                    auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
                    TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
                    //Eigen::VectorXd d = cg_solver.compute(H_proj + 1e-4 * TinyAD::identity<double>(x.size())).solve(-g);
                    Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
                    if (TinyAD::newton_decrement(d, g) < convergence_eps) {
                        //break;
                    }
                    x = TinyAD::line_search(x, d, f, g, func);

                    func.x_to_data(x, [&](int v_idx, const Eigen::VectorXd &p) {
                        if(v_idx >= Polygons.rows()) {
                            lagrangeMultipliers.row(v_idx-Polygons.rows()) = p;
                        } else {
                            Polygons.row(v_idx) = p;
                        }
                        // V.row(v_idx) = p(seq(0, 2));
                        // Normals.row(v_idx) = p(seq(3, 5));
                        //P.row(v_idx) = p;
                    });
                }
                else {

                    MatrixXd bc(b.size(), centers.cols());
                    for (int j = 0; j < b.size(); j++) {
                        bc.row(j) = centers.row(b(j));
                        for(int k = 0; k< conP.size(); k++) {
                           if(conP(k) == b(j)) {
                               bc.row(j) = constraints.row(k);
                               break;
                           }
                        }
                    }

                    custom_arap_solve(bc, arap_data, custom_data, U, rotations, Polygons);
                    for(int j = 0; j<centers.rows(); j++) {
                        //std::cout << rotations << std::endl;
                        Matrix3d rot = rotations.block<3, 3>(0, j * 3);


                        Eigen::Vector3d normal = originalPolygons.row(j).head(3);
                        normal = (rot * normal).normalized();
                        double d = normal.dot(U.row(j));

                        Eigen::VectorXd polygon(4);

                        polygon << normal(0), normal(1) , normal(2), d;
                        Polygons.row(j) = polygon;

                        //std::cout << rotations[j] << std::endl;
                    }

                }
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


    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) {
        if (redraw) {

            draw_face_mesh(viewer, Polygons);

            Eigen::RowVector3d point(x,y,z);
            //viewer.data().add_points(point, Eigen::RowVector3d(1, 0, 0));

            for(int i = 0; i< constraints.rows(); i++) {

                viewer.data().add_points(constraints.row(i), Eigen::RowVector3d(1, 0, 0));
            }
            //viewer.data().set_vertices(P);
            //viewer.core().align_camera_center(P);
            //viewer.core().camera_zoom = 2;
            {
                std::lock_guard<std::mutex> lock(m);
                constraints(1,0) = x;
                constraints(1,1) = y;
                constraints(1,2) = z;
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
    if (optimization_thread.joinable()) {
        optimization_thread.join();
    }

    return 0;
}
