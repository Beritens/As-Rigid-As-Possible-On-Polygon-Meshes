//
// Created by ben on 09.09.24.
//

#ifndef CUSTOM_DATA_H
#define CUSTOM_DATA_H
#include <map>

struct custom_data {
    std::map<int, std::map<int, Eigen::Vector3d> > sideVecs;
    std::map<int, std::map<int, std::vector<Eigen::Vector3d> > > cornerVecs;
};

#endif //CUSTOM_DATA_H

