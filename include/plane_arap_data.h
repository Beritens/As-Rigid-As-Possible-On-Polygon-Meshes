//
// Created by ben on 27.09.24.
//

#ifndef PLANE_ARAP_DATA_H
#define PLANE_ARAP_DATA_H

struct plane_arap_data {
    Eigen::MatrixXd L;
    Eigen::MatrixXd Polygons;
    Eigen::MatrixXd V;
    Eigen::VectorXi b;
    std::vector<int> positions;
    std::vector<std::vector<double> > cotanWeights;
};

#endif //PLANE_ARAP_DATA_H
