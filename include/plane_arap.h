//
// Created by brichter on 7/8/2024.
//

#ifndef PLANE_ARAP_H
#define PLANE_ARAP_H


#include <igl/arap.h>

#include "poly_mesh_data.h"
#include "plane_arap_data.h"
#include <TinyAD/ScalarFunction.hh>
#include "helper.h"


//TODO: create actual datastructure to work with meshes
bool plane_arap_precomputation(
    poly_mesh_data &mesh_data,
    plane_arap_data &data,
    const Eigen::VectorXi &b);

/// Conduct arap solve.
///
/// @param[in] bc  #b by dim list of boundary conditions
/// @param[in] data  struct containing necessary precomputation and parameters
/// @param[in,out] U  #V by dim initial guess
/// @param[out] rotations rotations for each face
///
/// \fileinfo
///
/// \note While the libigl guidelines require outputs to be of type
/// PlainObjectBase so that the user does not need to worry about allocating
/// memory for the output, in this case, the user is required to give an initial
/// guess and hence fix the size of the problem domain.
/// Taking a reference to MatrixBase in this case thus allows the user to provide e.g.
/// a map to the position data, allowing seamless interoperability with user-defined
/// datastructures without requiring a copy.
bool global_distance_step(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data);

void adjustPlaneNormals(
    poly_mesh_data &mesh_data,
    plane_arap_data &data);


void getRotations(
    poly_mesh_data &mesh_data,
    plane_arap_data &data);

TinyAD::ScalarFunction<4, double, long long> getFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data);

TinyAD::ScalarFunction<4, double, long long> getConstraintFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data,
    int index);

TinyAD::ScalarFunction<4, double, long long> getEdgeFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data);

TinyAD::ScalarFunction<3, double, long long> getBlockFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    plane_arap_data &data);


#endif //PLANE_ARAP_H
