//
// Created by ben on 27.09.24.
//

#ifndef PLANE_ARAP_DATA_H
#define PLANE_ARAP_DATA_H

struct edge {
    int rot;
    int a;
    int b;
    double w;
};

struct extra_grad_plane {
    int vertex;
    int plane;
};

struct plane_arap_data {
    Eigen::SparseMatrix<double> L;
    Eigen::MatrixXd Polygons;
    Eigen::MatrixXd V;
    Eigen::VectorXi b;
    Eigen::MatrixXd R;
    std::vector<edge> edges;
    std::vector<int> distPos;
    std::vector<std::vector<double> > cotanWeights;
    std::vector<int> conP;
    std::vector<extra_grad_plane> extra_grad_planes;
    std::vector<double> lagrangeMultipliers;
};

#endif //PLANE_ARAP_DATA_H
