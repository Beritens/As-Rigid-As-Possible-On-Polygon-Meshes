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
    Eigen::MatrixXd originalV;
    std::vector<std::vector<int> > F;
    Eigen::MatrixXi T;
    Eigen::MatrixXd Planes;
    std::vector<std::vector<int> > VertNeighbors;
    std::vector<std::set<int> > FaceNeighbors;
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
    // Eigen::Vector3<T> x = m.partialPivLu().solve(b);
    // direct inverse might be faster for small matrices
    Eigen::Vector3<T> x = m.inverse() * b;
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
    data.Planes = Eigen::MatrixXd(0, 4);
    for (int i = 0; i < data.F.size(); i++) {
        int size = data.F[i].size();
        data.Planes.conservativeResize(data.Planes.rows() + 1, 4);
        Eigen::Vector3d pointa = data.V.row(data.F[i][0]);
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        for (int j = 1; j < size; j++) {
            int k = ((j) % (size - 1)) + 1;
            Eigen::Vector3d pointb = data.V.row(data.F[i][j]);
            Eigen::Vector3d pointc = data.V.row(data.F[i][k]);
            Eigen::Vector3d a = pointb - pointa;
            Eigen::Vector3d b = pointc - pointa;
            normal += a.cross(b).normalized();
        }
        normal = normal.normalized();
        double dist = 0;
        for (int j = 0; j < size; j++) {
            dist += data.V.row(data.F[i][j]).dot(normal) / (double) size;
        }
        Eigen::Vector4d polygon;
        polygon << normal, dist;
        data.Planes.row(data.Planes.rows() - 1) = polygon;
    }
}

inline void precompute_poly_mesh(poly_mesh_data &data, Eigen::MatrixXd &V, std::vector<std::vector<int> > &F);

inline void calcNewV(poly_mesh_data &data);

inline void makeDual(poly_mesh_data &data) {
    Eigen::MatrixXd newV = data.originalV;
    std::vector<std::vector<int> > newF = data.F;
    bool mustRecompute = false;
    for (int i = 0; i < data.originalV.rows(); i++) {
        if (data.VertNeighbors[i].size() <= 3) {
            continue;
        }
        mustRecompute = true;
        std::map<int, int> verts;
        verts[data.VertNeighbors[i][0]] = i;
        std::vector<int> newFace;
        newFace.push_back(i);
        for (int j = 1; j < data.VertNeighbors[i].size(); j++) {
            newV.conservativeResize(newV.rows() + 1, 3);
            newV.row(newV.rows() - 1) = 0.999 * newV.row(i) + 0.001 * newV.row(data.VertNeighbors[i][j]);
            verts[data.VertNeighbors[i][j]] = newV.rows() - 1;
            newFace.push_back(newV.rows() - 1);
        }
        newV.row(i) = 0.999 * newV.row(i) + 0.001 * newV.row(data.VertNeighbors[i][0]);
        newF.push_back(newFace);
        for (auto f_idx: data.FaceNeighbors[i]) {
            int size = newF[f_idx].size();
            for (int j = 0; j < size; j++) {
                int next = (j + 1) % size;
                if (newF[f_idx][next] == i) {
                    int v1 = newF[f_idx][j];
                    int v2 = newF[f_idx][(next + 1) % size];
                    int n1 = verts[newF[f_idx][j]];
                    int n2 = verts[newF[f_idx][(next + 1) % size]];
                    newF[f_idx][next] = n1;
                    newF[f_idx].insert(newF[f_idx].begin() + next + 1, n2);

                    for (int k = 0; k < data.VertNeighbors[v1].size(); k++) {
                        if (data.VertNeighbors[v1][k] == i) {
                            data.VertNeighbors[v1][k] = n1;
                            break;
                        }
                    }
                    for (int k = 0; k < data.VertNeighbors[v2].size(); k++) {
                        if (data.VertNeighbors[v2][k] == i) {
                            data.VertNeighbors[v2][k] = n2;
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
    if (mustRecompute) {
        precompute_poly_mesh(data, newV, newF);
        calcNewV(data);
        data.originalV = data.V;
    }
}

inline void precompute_poly_mesh(poly_mesh_data &data, Eigen::MatrixXd &V, std::vector<std::vector<int> > &F) {
    data.V = V;
    data.originalV = V;
    data.F = F;
    calculatePolygons(data);
    std::map<int, std::set<int> > tempHoods;
    int triangleCount = 0;
    data.VertNeighbors.clear();
    data.FaceNeighbors.clear();
    data.VertNeighbors.resize(data.V.rows());
    data.FaceNeighbors.resize(data.V.rows());

    //get neighbours
    for (int i = 0; i < F.size(); i++) {
        const int size = F[i].size();
        triangleCount += size - 2;
        for (int j = 0; j < size; j++) {
            const int k = (j + 1) % size;
            tempHoods[F[i][j]].insert(F[i][k]);
            tempHoods[F[i][k]].insert(F[i][j]);
            data.FaceNeighbors[F[i][j]].insert(i);
        }
    }
    data.T = Eigen::MatrixXi::Zero(triangleCount, 3);

    for (int i = 0; i < V.rows(); i++) {
        int curr = *(tempHoods[i].begin());
        data.VertNeighbors[i].push_back(curr);

        while (data.VertNeighbors[i].size() < tempHoods[i].size()) {
            bool fin = false;
            for (auto v: tempHoods[i]) {
                bool alreadyIn = false;
                for (int existing: data.VertNeighbors[i]) {
                    if (v == existing) {
                        alreadyIn = true;
                        break;
                    }
                }
                if (alreadyIn) {
                    continue;
                }
                for (auto p1: data.FaceNeighbors[data.VertNeighbors[i][data.VertNeighbors[i].size() - 1]]) {
                    for (auto p2: data.FaceNeighbors[v]) {
                        if (p1 == p2) {
                            for (auto po: data.FaceNeighbors[i]) {
                                if (po == p1) {
                                    data.VertNeighbors[i].push_back(v);
                                    fin = true;
                                    break;
                                }
                            }
                            if (fin) {
                                break;
                            }
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
    makeDual(data);
}

inline std::vector<std::vector<int> > calculateTriangle(std::vector<Eigen::Vector3d> verts, Eigen::Vector3d normal) {
    int n = verts.size();
    Eigen::Vector3d a = (verts[1] - verts[0]).normalized();
    Eigen::Vector3d b = -(a.cross(normal)).normalized();
    Eigen::Matrix<double, 2, 3> M;
    M.row(0) = a;
    M.row(1) = b;
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
        bool weirdTri = false;
        for (int j = 0; j < 3; j++) {
            if (i * 3 + j >= indices.size()) {
                weirdTri = true;
                break;
            }
            if (indices[i * 3 + j] < 0 || indices[i * 3 + j] > verts.size()) {
                weirdTri = true;
            }
            if (weirdTri) {
                continue;
            }
            t.push_back(indices[i * 3 + j]);
        }
        if (weirdTri) {
            t = {0, 0, 0};
        }
        triangles.push_back(t);
    }
    return triangles;
}

inline void calculateTriangles(poly_mesh_data &mesh_data) {
    int t = 0;
    for (int i = 0; i < mesh_data.F.size(); i++) {
        int size = mesh_data.F[i].size();
        std::vector<Eigen::Vector3d> verts;
        for (int j = 0; j < size; j++) {
            int v_idx = mesh_data.F[i][j];
            verts.push_back(mesh_data.V.row(v_idx));
        }
        std::vector<std::vector<int> > triangles = calculateTriangle(verts, mesh_data.Planes.row(i).head(3));
        for (auto tri: triangles) {
            for (int j = 0; j < tri.size(); j++) {
                if (tri[j] < 0 || tri[j] >= size) {
                    continue;
                }
                mesh_data.T(t, j) = mesh_data.F[i][tri[j]];
            }
            t++;
        }
    }
}

inline void calcNewV(poly_mesh_data &data) {
    for (int i = 0; i < data.V.rows(); i++) {
        std::vector<Eigen::Vector4d> pols;
        for (auto pol: data.FaceNeighbors[i]) {
            pols.push_back(data.Planes.row(pol));
        }
        data.V.row(i) = getPoint<double>(pols[0], pols[1], pols[2]);
    }
}
#endif //POLY_MESH_DATA_H
