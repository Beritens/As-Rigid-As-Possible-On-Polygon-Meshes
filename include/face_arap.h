//
// Created by brichter on 7/8/2024.
//

#ifndef FACE_ARAP_H
#define FACE_ARAP_H


#include <igl/arap.h>

#include "poly_mesh_data.h"
#include "face_arap_data.h"
#include <TinyAD/ScalarFunction.hh>
#include "helper.h"

namespace std {
    template<int N, typename T, bool B>
    struct numeric_limits<TinyAD::Scalar<N, T, B> > {
        static constexpr bool is_specialized = true;

        static TinyAD::Scalar<N, T, B> min() noexcept {
            return TinyAD::Scalar<N, T, B>{
                std::numeric_limits<T>::min()
            };
        }

        static TinyAD::Scalar<N, T, B> max() noexcept {
            return TinyAD::Scalar<N, T, B>{
                std::numeric_limits<T>::max()
            };
        }

        static TinyAD::Scalar<N, T, B> lowest() noexcept {
            return TinyAD::Scalar<N, T, B>{
                std::numeric_limits<T>::lowest()
            };
        }

        static constexpr int digits = std::numeric_limits<T>::digits;
        static constexpr int digits10 = std::numeric_limits<T>::digits10;
        static constexpr int max_digits10 = std::numeric_limits<T>::max_digits10;

        static constexpr bool is_signed = std::numeric_limits<T>::is_signed;
        static constexpr bool is_integer = std::numeric_limits<T>::is_integer;
        static constexpr bool is_exact = std::numeric_limits<T>::is_exact;

        static constexpr bool has_infinity = std::numeric_limits<T>::has_infinity;
        static constexpr bool has_quiet_NaN = std::numeric_limits<T>::has_quiet_NaN;
        static constexpr bool has_signaling_NaN = std::numeric_limits<T>::has_signaling_NaN;
        static constexpr float_denorm_style has_denorm = std::numeric_limits<T>::has_denorm;
        static constexpr bool has_denorm_loss = std::numeric_limits<T>::has_denorm_loss;

        static TinyAD::Scalar<N, T, B> infinity() noexcept {
            return TinyAD::Scalar<N, T, B>{
                std::numeric_limits<T>::infinity()
            };
        }

        static TinyAD::Scalar<N, T, B> quiet_NaN() noexcept {
            return TinyAD::Scalar<N, T, B>{
                std::numeric_limits<T>::quiet_NaN()
            };
        }

        static TinyAD::Scalar<N, T, B> signaling_NaN() noexcept {
            return TinyAD::Scalar<N, T, B>{
                std::numeric_limits<T>::signaling_NaN()
            };
        }

        static TinyAD::Scalar<N, T, B> denorm_min() noexcept {
            return TinyAD::Scalar<N, T, B>{
                std::numeric_limits<T>::denorm_min()
            };
        }

        static constexpr bool is_iec559 = std::numeric_limits<T>::is_iec559;
        static constexpr bool is_bounded = std::numeric_limits<T>::is_bounded;
        static constexpr bool is_modulo = std::numeric_limits<T>::is_modulo;

        static constexpr bool traps = std::numeric_limits<T>::traps;
        static constexpr bool tinyness_before = std::numeric_limits<T>::tinyness_before;
        static constexpr float_round_style round_style = std::numeric_limits<T>::round_style;
    };
}

//TODO: create actual datastructure to work with meshes
bool face_arap_precomputation(
    poly_mesh_data &mesh_data,
    face_arap_data &data,
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
bool global_face_distance_step(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    face_arap_data &data);


void getFaceRotations(
    poly_mesh_data &mesh_data,
    face_arap_data &data);

TinyAD::ScalarFunction<4, double, long> getFaceFunction(
    const Eigen::MatrixXd &bc,
    poly_mesh_data &mesh_data,
    face_arap_data &data);
#endif
