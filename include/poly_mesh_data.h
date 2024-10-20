//
// Created by ben on 26.09.24.
//

#ifndef POLY_MESH_DATA_H
#define POLY_MESH_DATA_H
#include <map>
#include <set>
#include <vector>
#include <Eigen/Geometry>
#include "earcut.hpp"


struct poly_mesh_data {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXi T;
    int triangleCount;
    Eigen::MatrixXd Polygons;
    std::map<int, std::vector<int> > Hoods;
    std::map<int, std::set<int> > VertPolygons;
};

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

inline int faceSize(Eigen::VectorXi face) {
    int count = 0;
    for (int i = 0; i < face.size(); ++i) {
        if (face(i) != -1) {
            count++;
        }
    }
    return count;
}

inline void calculatePolygons(poly_mesh_data &data) {
    for (int i = 0; i < data.F.rows(); i++) {
        int size = faceSize(data.F.row(i));
        data.Polygons.conservativeResize(data.Polygons.rows() + 1, 4);
        Eigen::Vector3d pointa = data.V.row(data.F(i, 0));
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        for (int j = 1; j < size; j++) {
            int k = ((j) % (size - 1)) + 1;
            Eigen::Vector3d pointb = data.V.row(data.F(i, j));
            Eigen::Vector3d pointc = data.V.row(data.F(i, k));
            Eigen::Vector3d a = pointb - pointa;
            Eigen::Vector3d b = pointc - pointa;
            normal += a.cross(b).normalized();
        }
        normal = normal.normalized();
        double dist = pointa.dot(normal);
        Eigen::Vector4d polygon;
        polygon << normal, dist;
        data.Polygons.row(data.Polygons.rows() - 1) = polygon;
    }
}

inline void precompute_poly_mesh(poly_mesh_data &data, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    data.V = V;
    data.F = F;
    calculatePolygons(data);
    std::map<int, std::set<int> > tempHoods;
    data.triangleCount = 0;

    //get neighbours
    for (int i = 0; i < F.rows(); i++) {
        const int size = faceSize(F.row(i));
        data.triangleCount += size - 2;
        for (int j = 0; j < size; j++) {
            const int k = (j + 1) % size;
            tempHoods[F(i, j)].insert(F(i, k));
            tempHoods[F(i, k)].insert(F(i, j));
            data.VertPolygons[F(i, j)].insert(i);
        }
    }
    data.T = Eigen::MatrixXi(data.triangleCount, 3);

    for (int i = 0; i < V.size(); i++) {
        int curr = *(tempHoods[i].begin());
        data.Hoods[i].push_back(curr);

        while (data.Hoods[i].size() < tempHoods[i].size()) {
            bool fin = false;
            for (auto v: tempHoods[i]) {
                bool alreadyIn = false;
                for (int existing: data.Hoods[i]) {
                    if (v == existing) {
                        alreadyIn = true;
                        break;
                    }
                }
                if (alreadyIn) {
                    continue;
                }
                for (auto p1: data.VertPolygons[data.Hoods[i][data.Hoods[i].size() - 1]]) {
                    for (auto p2: data.VertPolygons[v]) {
                        if (p1 == p2) {
                            data.Hoods[i].push_back(v);
                            fin = true;
                            break;
                        }
                    }
                    if (fin) {
                        break;
                    }
                }
                if (fin) {
                    break;
                }
            }
        }
    }
}

inline std::vector<std::vector<int> > calculateTriangle(std::vector<Eigen::Vector3d> verts, Eigen::Vector3d normal) {
    int n = verts.size();
    Eigen::Vector3d a = (verts[1] - verts[0]).normalized();
    Eigen::Vector3d b = -(a.cross(normal)).normalized();
    Eigen::Matrix<double, 2, 3> M;
    M.row(0) = a;
    M.row(1) = b;
    double x[verts.size()];
    double y[verts.size()];
    using Point = std::array<double, 2>;
    std::vector<Point> poly;
    std::vector<std::vector<Point> > polygon;
    for (int i = 0; i < n; i++) {
        Eigen::Vector2d onPlane = M * (verts[i] - verts[0]);
        poly.push_back({onPlane(0), onPlane(1)});
    }
    polygon.push_back(poly);

    //int *tris = polygon_triangulate(n, x, y);
    std::vector<int> indices = mapbox::earcut<int>(polygon);
    std::vector<std::vector<int> > triangles;
    for (int i = 0; i < (n - 2); i++) {
        std::vector<int> t;
        for (int j = 0; j < 3; j++) {
            t.push_back(indices[i * 3 + j]);
        }
        triangles.push_back(t);
    }
    return triangles;
}

inline void calculateTriangles(poly_mesh_data &mesh_data) {
    int t = 0;
    for (int i = 0; i < mesh_data.F.rows(); i++) {
        int size = faceSize(mesh_data.F.row(i));
        std::vector<Eigen::Vector3d> verts;
        for (int j = 0; j < size; j++) {
            int v_idx = mesh_data.F(i, j);
            verts.push_back(mesh_data.V.row(v_idx));
        }
        try {
            std::vector<std::vector<int> > triangles = calculateTriangle(verts, mesh_data.Polygons.row(i).head(3));
            for (auto tri: triangles) {
                for (int j = 0; j < 3; j++) {
                    if (tri[j] < 0 || tri[j] >= size) {
                        continue;
                    }
                    mesh_data.T(t, j) = mesh_data.F(i, tri[j]);
                }
                t++;
            }
        } catch (...) {
            std::cout << "negative area" << std::endl;
            t += size - 2;
        }
    }
}

#endif //POLY_MESH_DATA_H
