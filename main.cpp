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
//Polygons around a vertex
Eigen::MatrixXi connections;
//center of polygons
Eigen::MatrixXd centers;
//group used for first method
Eigen::MatrixXi Groups;
//Neighbourhood of Polygon
std::map<int, std::vector<int> > Hoods;
//Neighbourhood Verts
std::map<int, std::vector<int> > HoodVerts;
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
poly_mesh_data mesh_data;

//0.947214,   0.58541,         0,
//0.8090169943749473, -0.8090169943749473, 0.8090169943749473;
// double x = 0.8090169943749473;
// double y = -0.8090169943749473;
// double z = 0.8090169943749473;
//
double x = 0;
double y = 0;
double z = 0;
bool autodiff = true;

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
        Eigen::Vector3d p = V.row(polyF(face, j));
        center += p * (1.0 / (size * 1.0));
    }
    centers.row(face) = center;
}

void calculateHood(Eigen::MatrixXi polyF, int face) {
    std::vector<int> neighbours;
    Eigen::Vector3d center;
    int size = faceSize(polyF.row(face));

    //TODO: find faces that are also connected to last faces (makes finding connections easier)
    int i = 0;
    while (i < polyF.rows()) {
        bool giveUp = i == face;
        for (int j = 0; j < neighbours.size(); j++) {
            if (i == neighbours[j]) {
                giveUp = true;
            }
        }
        if (giveUp) {
            i++;
            continue;
        }
        int otherFaceSize = faceSize(polyF.row(i));
        int lastFaceSize = -1;
        if (neighbours.size() > 0) {
            lastFaceSize = faceSize(polyF.row(neighbours[neighbours.size() - 1]));
        }
        int count = 0;
        int count2 = 0;

        for (int j = 0; j < otherFaceSize; j++) {
            for (int k = 0; k < size; k++) {
                if (polyF(face, k) == polyF(i, j)) {
                    count++;
                    if (count >= 2) {
                        break;
                    }
                }
            }
            for (int k = 0; k < lastFaceSize; k++) {
                if (polyF(neighbours[neighbours.size() - 1], k) == polyF(i, j)) {
                    count2++;
                    if (count2 >= 2) {
                        break;
                    }
                }
            }
            if (count >= 2 && (count2 >= 2 || lastFaceSize < 0)) {
                break;
            }
        }
        if (count >= 2 && (count2 >= 2 || lastFaceSize < 0)) {
            neighbours.push_back(i);
            i = 0;
        } else {
            i++;
        }
    }
    Hoods[face] = neighbours;
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
        calculateHood(polyF, i);
    }
    for (int i = 0; i < V.rows(); i++) {
        setConnectivityForOneVertex(polyF, i);
    }
    //basically just faces, but making sure they are in the right order
    for (int i = 0; i < Hoods.size(); i++) {
        std::vector<int> neighbours = Hoods[i];
        std::vector<int> verts;
        for (int j = 0; j < neighbours.size(); j++) {
            int k = (j + 1) % neighbours.size();
            std::vector<Eigen::VectorXi> faces;
            faces.push_back(polyF.row(i));
            faces.push_back(polyF.row(neighbours[j]));
            faces.push_back(polyF.row(neighbours[k]));
            std::set<int> shared_verts;
            std::set<int> shared_verts_temp;

            for (int v: faces[0]) {
                if (v < 0) {
                    break;
                }
                shared_verts.insert(v);
            }
            for (int idx = 1; idx < 3; idx++) {
                for (int v: faces[idx]) {
                    if (v < 0) {
                        break;
                    }
                    if (shared_verts.count(v) > 0) {
                        shared_verts_temp.insert(v);
                    }
                }
                shared_verts = shared_verts_temp;
                shared_verts_temp = {};
            }
            verts.push_back(*shared_verts.begin());
        }
        HoodVerts[i] = verts;
    }
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


    igl::opengl::glfw::Viewer viewer;
    draw_face_mesh(viewer, Polygons);

    Matrix3d rotationTest = getRotation<double>(V, V);
    std::cout << rotationTest << std::endl;


    Eigen::VectorXi conP(2);
    // conP << 0, 12;
    conP << 0, 39;
    Eigen::MatrixXd lagrangeMultipliers = Eigen::MatrixXd::Zero(conP.size(), 4);
    Eigen::MatrixXd constraints(conP.size(), 3);
    for (int i = 0; i < conP.size(); i++) {
        if (autodiff) {
            constraints.row(i) = V.row(conP(i));
        } else {
            constraints.row(i) = centers.row(conP(i));
        }
    }
    x = constraints(1, 0);
    y = constraints(1, 1);
    z = constraints(1, 2);

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
            for (int i = 0; i < conP.size(); i++) {
                b(i) = conP(i);
            }

            igl::ARAPData arap_data;
            plane_arap_data plane_arap_data;
            arap_data.max_iter = 1;
            if (!autodiff) {
                custom_arap_precomputation(centers, connections, centers.cols(), b, arap_data, custom_data, Polygons, V,
                                           polyF);
            }

            precompute_poly_mesh(mesh_data, V, polyF);

            plane_arap_precomputation(mesh_data, plane_arap_data, b);


            for (int i = 0; i < connections.rows(); i++) {
                VertsMap(i, 0) = i;

                Vector4d poly1 = Polygons.row(connections(i, 0));
                Vector4d poly2 = Polygons.row(connections(i, 1));
                Vector4d poly3 = Polygons.row(connections(i, 2));

                Verts.row(i) = getPoint(poly1, poly2, poly3);
                Vector3i polies;
                polies << connections(i, 0), connections(i, 1), connections(1, 2);
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
            auto func = TinyAD::scalar_function<4>(TinyAD::range(Polygons.rows()));


            // bool penalty = true;
            // func.add_elements<4>(TinyAD::range(conP.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
            //     using T = TINYAD_SCALAR_TYPE(element);
            //     Eigen::Index f_idx = element.handle;
            //     //Eigen::Vector4<T> lagrangeM = element.variables(f_idx + Polygons.rows());
            //     std::vector<Eigen::Vector4<T> > vecs;
            //     int i = 0;
            //     for (auto p: mesh_data.VertPolygons[conP(f_idx)]) {
            //         Eigen::Vector4<T> pol;
            //         pol = element.variables(p);
            //         // pol.head(3) = element.variables(p);
            //         // pol(3) = mesh_data.Polygons(p, 3);
            //         vecs.push_back(pol);
            //         i++;
            //     }
            //     Eigen::Vector3<T> point = getPoint<T>(vecs[0], vecs[1], vecs[2]);
            //     Eigen::Vector3d target = constraints.row(f_idx);
            //     return 0 * (target - point).squaredNorm();
            // });
            //
            // bool face = false;
            // if (face) {
            //     func.add_elements<7>(TinyAD::range(Hoods.size()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
            //         using T = TINYAD_SCALAR_TYPE(element);
            //
            //         Eigen::Index f_idx = element.handle;
            //         std::vector<int> neighbours = Hoods[f_idx];
            //         Eigen::MatrixX<T> corners(neighbours.size(), 3);
            //         for (int i = 0; i < neighbours.size(); i++) {
            //             int j = (i + 1) % neighbours.size();
            //             Eigen::Vector4<T> p1 = element.variables(f_idx);
            //             Eigen::Vector4<T> p2 = element.variables(neighbours[i]);
            //             Eigen::Vector4<T> p3 = element.variables(neighbours[j]);
            //             Eigen::Vector3<T> point = getPoint<T>(p1, p2, p3);
            //             corners.row(i) = point;
            //         }
            //         Eigen::Vector3<T> center;
            //         center << 0, 0, 0;
            //         for (int i = 0; i < corners.rows(); i++) {
            //             center += corners.row(i) / corners.rows();
            //         }
            //
            //
            //         Eigen::MatrixXd V1(corners.rows(), 3);
            //         Eigen::MatrixX<T> V2(corners.rows(), 3);
            //
            //         for (int i = 0; i < corners.rows(); i++) {
            //             V2.row(i) = corners.row(i) - center.transpose();
            //         }
            //
            //         for (int i = 0; i < corners.rows(); i++) {
            //             Eigen::Vector3d vert = Verts.row(HoodVerts[f_idx][i]);
            //             Eigen::Vector3d oldCenter = centers.row(f_idx);
            //             V1.row(i) = vert - oldCenter;
            //         }
            //
            //         Eigen::Matrix3<T> Rot = getRotation<T>(V1, V2);
            //
            //         T returnValue = 0;
            //         for (int i = 0; i < V1.rows(); i++) {
            //             returnValue += (V1.row(i).transpose() - (Rot * V2.row(i).transpose())).squaredNorm();
            //         }
            //         return returnValue;
            //     });
            // } else {
            func.add_elements<9>(TinyAD::range(mesh_data.V.rows()),
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
                                         // bool con = false;
                                         // for (int i = 0; i < conP.size(); i++) {
                                         //     if (neighbor == conP(i)) {
                                         //         Eigen::Vector3<T> conVert = constraints.row(i);
                                         //         points.push_back(conVert);
                                         //         con = true;
                                         //     }
                                         // }
                                         // if (con) {
                                         //     continue;
                                         // }
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
                i++;
                if (autodiff) {
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
                    // Eigen::VectorXd d = cg_solver.compute(H_proj + 1e-6 * TinyAD::identity<double>(x.size())).solve(-g);
                    Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
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
                } else {
                    MatrixXd bc(b.size(), centers.cols());
                    for (int j = 0; j < b.size(); j++) {
                        bc.row(j) = centers.row(b(j));
                        for (int k = 0; k < conP.size(); k++) {
                            if (conP(k) == b(j)) {
                                bc.row(j) = constraints.row(k);
                                break;
                            }
                        }
                    }

                    custom_arap_solve(bc, arap_data, custom_data, U, rotations, originalPolygons);
                    for (int j = 0; j < centers.rows(); j++) {
                        //std::cout << rotations << std::endl;
                        Matrix3d rot = rotations.block<3, 3>(0, j * 3);


                        Eigen::Vector3d normal = originalPolygons.row(j).head(3);
                        normal = (rot * normal).normalized();
                        double d = normal.dot(U.row(j));

                        Eigen::VectorXd polygon(4);

                        polygon << normal(0), normal(1), normal(2), d;
                        Polygons.row(j) = polygon;

                        //std::cout << rotations[j] << std::endl;
                    }
                } {
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

            Eigen::RowVector3d point(x, y, z);
            //viewer.data().add_points(point, Eigen::RowVector3d(1, 0, 0));

            for (int i = 0; i < constraints.rows(); i++) {
                viewer.data().add_points(constraints.row(i), Eigen::RowVector3d(1, 0, 0));
            }
            //viewer.data().set_vertices(P);
            //viewer.core().align_camera_center(P);
            //viewer.core().camera_zoom = 2;
            {
                std::lock_guard<std::mutex> lock(m);
                constraints(1, 0) = x;
                constraints(1, 1) = y;
                constraints(1, 2) = z;
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
