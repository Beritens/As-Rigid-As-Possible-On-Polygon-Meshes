//
// Created by ben on 14.10.24.
//

#ifndef HELPER_H
#define HELPER_H
#include "TinyAD/Detail/EigenVectorTypedefs.hh"

template<class T>
Eigen::Matrix3<T> getRotation(Eigen::MatrixX<T> v1, Eigen::MatrixX<T> v2) {
    Eigen::Matrix3<T> S = v2.transpose() * v1;
    Eigen::JacobiSVD<Eigen::Matrix3<T> > svd(S, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::MatrixX<T> U = svd.matrixU();
    Eigen::MatrixX<T> V = svd.matrixV();
    Eigen::Matrix3<T> Rot = U * V.transpose();

    if (Rot.determinant() < 0) {
        Eigen::Matrix3<T> I = Eigen::Matrix3<T>::Identity();
        I(2, 2) = -1;
        Rot = U * I * V.transpose();
    }

    return Rot;
}

inline double getAngle(Eigen::Vector3d a, Eigen::Vector3d b) {
    return acos(a.dot(b) / (a.norm() * b.norm()));
}
#endif //HELPER_H
