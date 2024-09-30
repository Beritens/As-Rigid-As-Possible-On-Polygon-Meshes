//
// Created by ben on 26.09.24.
//

#ifndef POLY_MESH_DATA_H
#define POLY_MESH_DATA_H
#include <map>
#include <set>
#include <vector>
#include <Eigen/Geometry>


struct poly_mesh_data {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd Polygons;
    std::map<int, std::set<int> > Hoods;
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
        data.Polygons.conservativeResize(data.Polygons.rows() + 1, 4);
        Eigen::Vector3d pointa = data.V.row(data.F(i, 0));
        Eigen::Vector3d pointb = data.V.row(data.F(i, 1));
        Eigen::Vector3d pointc = data.V.row(data.F(i, 2));
        Eigen::Vector3d a = pointb - pointa;
        Eigen::Vector3d b = pointc - pointa;
        Eigen::Vector3d normal = a.cross(b).normalized();
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

    //get neighbours
    for (int i = 0; i < F.rows(); i++) {
        const int size = faceSize(F.row(i));
        for (int j = 0; j < size; j++) {
            const int k = (j + 1) % size;
            data.Hoods[F(i, j)].insert(F(i, k));
            data.Hoods[F(i, k)].insert(F(i, j));
            data.VertPolygons[F(i, j)].insert(i);
        }
    }
}

#endif //POLY_MESH_DATA_H