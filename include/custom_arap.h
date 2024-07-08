//
// Created by brichter on 7/8/2024.
//

#ifndef CUSTOM_ARAP_H
#define CUSTOM_ARAP_H




#include <igl/arap.h>



  /// Parameters and precomputed values for custom arap solver.
  ///
  /// \fileinfo


  /// Compute necessary information to start using an ARAP deformation using
  /// local-global solver as described in "As-rigid-as-possible surface
  /// modeling" [Sorkine and Alexa 2007].
  ///
  /// @param[in] V  #V by dim list of mesh positions
  /// @param[in] F  #F by simplex-size list of triangle|tet indices into V
  /// @param[in] dim  dimension being used at solve time. For deformation usually dim =
  ///    V.cols(), for surface parameterization V.cols() = 3 and dim = 2
  /// @param[in] b  #b list of "boundary" fixed vertex indices into V
  /// @param[out] data  struct containing necessary precomputation
  /// @return whether initialization succeeded
  ///
  /// \fileinfo
  template <
    typename DerivedV,
    typename DerivedF,
    typename Derivedb>
  bool custom_arap_precomputation(
      const Eigen::MatrixBase<DerivedV>& V,
      const Eigen::MatrixBase<DerivedF>& F,
      const int dim,
      const Eigen::MatrixBase<Derivedb>& b,
      igl::ARAPData& data);
  /// Conduct arap solve.
  ///
  /// @param[in] bc  #b by dim list of boundary conditions
  /// @param[in] data  struct containing necessary precomputation and parameters
  /// @param[in,out] U  #V by dim initial guess
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
  template <
    typename Derivedbc,
    typename DerivedU>
  bool custom_arap_solve(
      const Eigen::MatrixBase<Derivedbc>& bc,
      igl::ARAPData& data,
      Eigen::MatrixBase<DerivedU>& U,
      Eigen::MatrixXd& rotations);


#endif //CUSTOM_ARAP_H