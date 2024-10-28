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

struct plane_arap_data {
    Eigen::MatrixXd L;
    Eigen::MatrixXd Polygons;
    Eigen::MatrixXd V;
    Eigen::VectorXi b;
    Eigen::MatrixXd R;
    std::vector<edge> edges;
    std::vector<int> distPos;
    std::vector<std::vector<double> > cotanWeights;
    std::vector<int> conP;
};

#endif //PLANE_ARAP_DATA_H
