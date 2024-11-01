//
// Created by ben on 27.09.24.
//

#ifndef FACE_ARAP_DATA_H
#define FACE_ARAP_DATA_H

struct face_arap_data {
    Eigen::SparseMatrix<double> L;
    Eigen::MatrixXd Polygons;
    Eigen::MatrixXd V;
    Eigen::VectorXi b;
    Eigen::MatrixXd simpleB;
    Eigen::SparseMatrix<double> B;
    Eigen::SparseMatrix<double> Bt;
    Eigen::MatrixXd R;
    std::vector<std::vector<std::vector<int> > > triangles;
    std::vector<std::vector<std::vector<double> > > cotanWeights;
    std::vector<int> conP;
    std::vector<int> distPos;
};

#endif //PLANE_ARAP_DATA_H
