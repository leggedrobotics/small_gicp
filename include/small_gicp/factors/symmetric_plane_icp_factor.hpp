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
  struct Setting {};  // No parameters, interface unchanged

  explicit SymmetricPointToPlaneICPFactor(const Setting& = Setting())
      : target_index(std::numeric_limits<size_t>::max()),
        source_index(std::numeric_limits<size_t>::max()) {}

  template <typename TargetPointCloud,
            typename SourcePointCloud,
            typename TargetTree,
            typename CorrespondenceRejector>
  bool linearize(const TargetPointCloud&  target,
                 const SourcePointCloud&  source,
                 const TargetTree&        target_tree,
                 const Eigen::Isometry3d& T,
                 size_t                   source_idx,
                 const CorrespondenceRejector& rejector,
                 Eigen::Matrix<double, 6, 6>* H,
                 Eigen::Matrix<double, 6, 1>* b,
                 double*                   e)
  {
    constexpr size_t k_neighbors = 5;              
    constexpr double degenerate_threshold = 1e-6;  // Threshold for ‖n_s + n_t‖²

    source_index = source_idx;

    const Eigen::Vector4d p_s_v4    = traits::point(source, source_idx);
    const Eigen::Vector4d p_s_v4_tr = T * p_s_v4;
    const Eigen::Vector3d p_s       = p_s_v4.template head<3>();
    const Eigen::Vector3d p_s_tr    = p_s_v4_tr.template head<3>();
    const Eigen::Vector3d n_s       = traits::normal(source, source_idx).template head<3>();

    std::array<size_t, k_neighbors>  k_indices;
    std::array<double, k_neighbors>  k_sq_dists;

    if (traits::knn_search(target_tree, p_s_v4_tr, k_neighbors, k_indices.data(), k_sq_dists.data()) != k_neighbors) {
      return false;
    }

    Eigen::Matrix<double, 6, 6> H_acc = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b_acc = Eigen::Matrix<double, 6, 1>::Zero();
    double e_acc = 0.0;
    bool valid = false;

    const Eigen::Matrix3d& R = T.linear();

    for (size_t i = 0; i < k_neighbors; ++i) {
      size_t k_index = k_indices[i];
      double sq_dist = k_sq_dists[i];

      if (rejector(target, source, T, k_index, source_idx, sq_dist)) continue;

      this->target_index = k_index;

      const Eigen::Vector3d p_t = traits::point(target, k_index).template head<3>();
      Eigen::Vector3d n_t       = traits::normal(target, k_index).template head<3>();

      // ---- Mathematically consistent flipping: align n_t with n_s
      if (n_s.dot(n_t) < 0.0) n_t = -n_t;

      // ---- Averaged normal, only if not degenerate
      Eigen::Vector3d n_avg = n_s + n_t;
      // double norm2 = n_avg.squaredNorm();

      // if (norm2 < degenerate_threshold) continue;   // skip nearly opposite normals

      n_avg.normalize();

      // ---- Residual and Jacobian, using correct right-mult. convention
      const Eigen::Vector3d r = p_s_tr - p_t;
      double err = n_avg.dot(r);

      Eigen::Matrix<double, 1, 6> J;
      J.leftCols<3>()  = n_avg.transpose() * (-R * skew(p_s));
      J.rightCols<3>() = n_avg.transpose() *  R;

      H_acc.noalias() += J.transpose() * J;
      b_acc.noalias() += J.transpose() * err;
      e_acc          += 0.5 * err * err;

      valid = true;
    }

    if (valid) {
      *H = H_acc;
      *b = b_acc;
      *e = e_acc;
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

    const Eigen::Vector3d p_s_tr = (T * traits::point(source, source_index)).template head<3>();
    const Eigen::Vector3d p_t    = traits::point(target, target_index).template head<3>();
    const Eigen::Vector3d r      = p_s_tr - p_t;

    const Eigen::Vector3d n_s = traits::normal(source, source_index).template head<3>();
    const Eigen::Vector3d n_t_unflipped = traits::normal(target, target_index).template head<3>();
    Eigen::Vector3d n_t = n_t_unflipped;

    // Consistent flipping for error computation as well
    if (n_s.dot(n_t) < 0.0) n_t = -n_t;
    Eigen::Vector3d n_avg = n_s + n_t;
    double norm2 = n_avg.squaredNorm();
    if (norm2 < 1e-6) return 0.0;
    n_avg.normalize();

    double err = n_avg.dot(r);

    return 0.5 * err * err;
  }

  bool inlier() const { return target_index != std::numeric_limits<size_t>::max(); }
  void reset_inlier() { target_index = std::numeric_limits<size_t>::max();        }

  size_t target_index;
  size_t source_index;
};


}  // namespace small_gicp
