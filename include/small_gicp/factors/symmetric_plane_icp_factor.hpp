// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/// @brief Symmetric point-to-plane per-point error factor.
struct SymmetricPointToPlaneICPFactor {
  struct Setting {};

  SymmetricPointToPlaneICPFactor(const Setting& = Setting())
      : target_index(std::numeric_limits<size_t>::max()),
        source_index(std::numeric_limits<size_t>::max()) {}

  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector>
  bool linearize(const TargetPointCloud& target,
                 const SourcePointCloud& source,
                 const TargetTree& target_tree,
                 const Eigen::Isometry3d& T,
                 size_t source_index,
                 const CorrespondenceRejector& rejector,
                 Eigen::Matrix<double, 6, 6>* H,
                 Eigen::Matrix<double, 6, 1>* b,
                 double* e) {
    this->source_index = source_index;
    constexpr size_t k_neighbors = 3;
    std::array<size_t, k_neighbors> k_indices;
    std::array<double, k_neighbors> k_sq_dists;

    const Eigen::Vector4d p_s_v4 = traits::point(source, source_index);
    const Eigen::Vector4d transed_source_pt = T * p_s_v4;
    const Eigen::Vector3d p_s = p_s_v4.template head<3>();
    const Eigen::Vector3d p_s_trans = transed_source_pt.template head<3>();
    const Eigen::Vector3d n_s = traits::normal(source, source_index).template head<3>();

    if (traits::knn_search(target_tree, transed_source_pt, k_neighbors, k_indices.data(), k_sq_dists.data()) != k_neighbors) {
      return false;
    }

    Eigen::Matrix<double, 6, 6> H_accum = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b_accum = Eigen::Matrix<double, 6, 1>::Zero();
    double e_accum = 0.0;
    bool valid = false;

    for (size_t i = 0; i < k_neighbors; ++i) {
      size_t k_index = k_indices[i];
      double sq_dist = k_sq_dists[i];

      if (rejector(target, source, T, k_index, source_index, sq_dist)) continue;

      this->target_index = k_index;

      const Eigen::Vector3d p_t = traits::point(target, k_index).template head<3>();
      const Eigen::Vector3d n_t = traits::normal(target, k_index).template head<3>();
      const Eigen::Vector3d r = p_s_trans - p_t;
      const Eigen::Vector2d err_vec(n_s.dot(r), n_t.dot(r));

      Eigen::Matrix<double, 3, 6> J_r;
      J_r.leftCols<3>() = -T.linear() * skew(p_s);
      J_r.rightCols<3>() = T.linear();

      Eigen::Matrix<double, 2, 6> J;
      J.row(0) = n_s.transpose() * J_r;
      J.row(1) = n_t.transpose() * J_r;

      H_accum.noalias() += J.transpose() * J;
      b_accum.noalias() += J.transpose() * err_vec;
      e_accum += 0.5 * err_vec.squaredNorm();

      valid = true;
    }

    if (valid) {
      *H = H_accum;
      *b = b_accum;      
      *e = e_accum;
    }

    return valid;
  }

  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud& target,
               const SourcePointCloud& source,
               const Eigen::Isometry3d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    const Eigen::Vector3d p_s_trans = (T * traits::point(source, source_index)).template head<3>();
    const Eigen::Vector3d p_t = traits::point(target, target_index).template head<3>();
    const Eigen::Vector3d r = p_s_trans - p_t;

    const Eigen::Vector3d n_s = traits::normal(source, source_index).template head<3>();
    const Eigen::Vector3d n_t = traits::normal(target, target_index).template head<3>();

    const double e0 = n_s.dot(r);
    const double e1 = n_t.dot(r);

    return 0.5 * (e0 * e0 + e1 * e1);
  }

  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }
  void reset_inlier() { target_index = std::numeric_limits<size_t>::max(); }

  size_t target_index;
  size_t source_index;
};

// /// @brief Symmetric point-to-plane per-point error factor.
// struct SymmetricPointToPlaneICPFactor {
//   struct Setting {};

//   SymmetricPointToPlaneICPFactor(const Setting& = Setting())
//       : target_index(std::numeric_limits<size_t>::max()),
//         source_index(std::numeric_limits<size_t>::max()) {}

//   template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree, typename CorrespondenceRejector>
//   bool linearize(const TargetPointCloud& target,
//                  const SourcePointCloud& source,
//                  const TargetTree& target_tree,
//                  const Eigen::Isometry3d& T,
//                  size_t source_index,
//                  const CorrespondenceRejector& rejector,
//                  Eigen::Matrix<double, 6, 6>* H,
//                  Eigen::Matrix<double, 6, 1>* b,
//                  double* e) {
//     this->source_index = source_index;
//     this->target_index = std::numeric_limits<size_t>::max();

//     const Eigen::Vector4d p_s_v4 = traits::point(source, source_index);
//     const Eigen::Vector4d transed_source_pt = T * p_s_v4;

//     size_t k_index;
//     double k_sq_dist;
//     if (!traits::nearest_neighbor_search(target_tree, transed_source_pt, &k_index, &k_sq_dist) ||
//         rejector(target, source, T, k_index, source_index, k_sq_dist)) {
//       return false;
//     }

//     target_index = k_index;

//     const Eigen::Vector3d p_s = p_s_v4.template head<3>();
//     const Eigen::Vector3d n_s = traits::normal(source, source_index).template head<3>();
//     const Eigen::Vector3d p_t = traits::point(target, target_index).template head<3>();
//     const Eigen::Vector3d n_t = traits::normal(target, target_index).template head<3>();

//     const Eigen::Vector3d p_s_trans = transed_source_pt.template head<3>();
//     const Eigen::Vector3d r = p_s_trans - p_t;

//     const Eigen::Vector2d err_vec(n_s.dot(r), n_t.dot(r));

//     Eigen::Matrix<double, 3, 6> J_r;
//     J_r.leftCols<3>() = -T.linear() * skew(p_s);  // dR * skew
//     J_r.rightCols<3>() = T.linear();              // dT
    
//     Eigen::Matrix<double, 2, 6> J;
//     J.row(0) = n_s.transpose() * J_r;
//     J.row(1) = n_t.transpose() * J_r;

//     *H = J.transpose() * J;
//     *b = J.transpose() * err_vec;
//     *e = 0.5 * err_vec.squaredNorm();

//     return true;
//   }

//   template <typename TargetPointCloud, typename SourcePointCloud>
//   double error(const TargetPointCloud& target,
//                const SourcePointCloud& source,
//                const Eigen::Isometry3d& T) const {
//     if (target_index == std::numeric_limits<size_t>::max()) {
//       return 0.0;
//     }

//     const Eigen::Vector3d p_s_trans = (T * traits::point(source, source_index)).template head<3>();
//     const Eigen::Vector3d p_t = traits::point(target, target_index).template head<3>();
//     const Eigen::Vector3d r = p_s_trans - p_t;

//     const Eigen::Vector3d n_s = traits::normal(source, source_index).template head<3>();
//     const Eigen::Vector3d n_t = traits::normal(target, target_index).template head<3>();

//     const double e0 = n_s.dot(r);
//     const double e1 = n_t.dot(r);

//     return 0.5 * (e0 * e0 + e1 * e1);
//   }

//   bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }
//   void reset_inlier() { target_index = std::numeric_limits<size_t>::max(); }

//   size_t target_index;
//   size_t source_index;
// };

}  // namespace small_gicp
