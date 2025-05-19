// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <iostream>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/registration/registration_result.hpp>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
namespace small_gicp {

struct SolutionRemappingHandler {
  bool enabled = false;
  double threshold = 250;

  bool has_degenerate = false;
  Eigen::Matrix<double, 6, 6> projection = Eigen::Matrix<double, 6, 6>::Identity();

  void compute(const Eigen::Matrix<double, 6, 6>& H) {
    has_degenerate = false;
    enabled = false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(H);
    const auto& eigvals = solver.eigenvalues();
    const auto& eigvecs = solver.eigenvectors();

    Eigen::Matrix<double, 6, 6> eigvecs_copy = eigvecs;

    int degenerate_index = -1;
    for (int i = 0; i < 6; ++i) {
      if (eigvals(i) < threshold) {
        eigvecs_copy.col(i).setZero();
        has_degenerate = true;
        degenerate_index = i;
      }
    }

    projection = eigvecs.transpose().inverse() * eigvecs_copy.transpose();
    if (has_degenerate) {
      enabled = true;
      std::cerr << "\033[1;31mWarning: Eigenvector " << degenerate_index << " is degenerate!\033[0m" << std::endl;
    }
  }

  void apply(Eigen::Matrix<double, 6, 1>& delta) const {
    if (enabled) {
      delta = projection * delta;
    }
  }
};

/// @brief GaussNewton optimizer
struct GaussNewtonOptimizer {
  GaussNewtonOptimizer() : verbose(false), max_iterations(20), lambda(1e-6) {}

  template <
    typename TargetPointCloud,
    typename SourcePointCloud,
    typename TargetTree,
    typename CorrespondenceRejector,
    typename TerminationCriteria,
    typename Reduction,
    typename Factor,
    typename GeneralFactor>
  RegistrationResult optimize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const CorrespondenceRejector& rejector,
    const TerminationCriteria& criteria,
    Reduction& reduction,
    const Eigen::Isometry3d& init_T,
    std::vector<Factor>& factors,
    GeneralFactor& general_factor) const {
    //
    if (verbose) {
      std::cout << "--- GN optimization ---" << std::endl;
    }

    RegistrationResult result(init_T);
    for (int i = 0; i < max_iterations && !result.converged; i++) {
      // Linearize
      auto [H, b, e] = reduction.linearize(target, source, target_tree, rejector, result.T_target_source, factors);
      general_factor.update_linearized_system(target, source, target_tree, result.T_target_source, &H, &b, &e);

      // Solve linear system
      const Eigen::Matrix<double, 6, 1> delta = (H + lambda * Eigen ::Matrix<double, 6, 6>::Identity()).ldlt().solve(-b);

      if (verbose) {
        std::cout << "iter=" << i << " e=" << e << " lambda=" << lambda << " dt=" << delta.tail<3>().norm() << " dr=" << delta.head<3>().norm() << std::endl;
      }

      result.converged = criteria.converged(delta);
      result.T_target_source = result.T_target_source * se3_exp(delta);
      result.iterations = i;
      result.H = H;
      result.b = b;
      result.error = e;
    }

    result.num_inliers = std::count_if(factors.begin(), factors.end(), [](const auto& factor) { return factor.inlier(); });

    return result;
  }

  bool verbose;        ///< If true, print debug messages
  int max_iterations;  ///< Max number of optimization iterations
  double lambda;       ///< Damping factor (Increasing this makes optimization slow but stable)
};

/// @brief LevenbergMarquardt optimizer
struct LevenbergMarquardtOptimizer {
  LevenbergMarquardtOptimizer() : verbose(false), max_iterations(20), max_inner_iterations(10), init_lambda(1e-3), lambda_factor(10.0) {}

  template <
    typename TargetPointCloud,
    typename SourcePointCloud,
    typename TargetTree,
    typename CorrespondenceRejector,
    typename TerminationCriteria,
    typename Reduction,
    typename Factor,
    typename GeneralFactor>
  RegistrationResult optimize(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const CorrespondenceRejector& rejector,
    const TerminationCriteria& criteria,
    Reduction& reduction,
    const Eigen::Isometry3d& init_T,
    std::vector<Factor>& factors,
    GeneralFactor& general_factor) const {
    //
    if (verbose) {
      std::cout << "--- LM optimization ---" << std::endl;
    }

    double lambda = init_lambda;
    RegistrationResult result(init_T);
    SolutionRemappingHandler remap_handler;
    for (int i = 0; i < max_iterations && !result.converged; i++) {
      // Linearize
      auto [H, b, e] = reduction.linearize(target, source, target_tree, rejector, result.T_target_source, factors);
      general_factor.update_linearized_system(target, source, target_tree, result.T_target_source, &H, &b, &e);
      remap_handler.compute(H);

      // Lambda iteration
      bool success = false;
      for (int j = 0; j < max_inner_iterations; j++) {
        // Apply damping
        Eigen::Matrix<double, 6, 6> H_damped = H + lambda * Eigen::Matrix<double, 6, 6>::Identity();

        // Block-diagonal preconditioner (rotation/translation split)
        Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Identity();
        for (int k = 0; k < 3; ++k) {
          double r = std::max(H_damped(k, k), 1e-12);
          double t = std::max(H_damped(k + 3, k + 3), 1e-12);
          P(k, k) = 1.0 / std::sqrt(r);
          P(k + 3, k + 3) = 1.0 / std::sqrt(t);
        }

        // Preconditioned system
        Eigen::Matrix<double, 6, 6> A = P * H_damped * P;
        Eigen::Matrix<double, 6, 1> rhs = -P * b;

        // Solve A * delta_hat = rhs
        Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix<double, 6, 1> delta_hat;
        {
          const double tol = 1e-12;
          Eigen::Matrix<double, 6, 6> S_inv = Eigen::Matrix<double, 6, 6>::Zero();
          const auto& sing = svd.singularValues();
          for (int k = 0; k < 6; ++k)
            if (sing[k] > tol) S_inv(k, k) = 1.0 / sing[k];

          delta_hat = svd.matrixV() * S_inv * svd.matrixU().transpose() * rhs;
        }

        Eigen::Matrix<double, 6, 1> delta = P * delta_hat;
        remap_handler.apply(delta);
        // Validate new solution
        const Eigen::Isometry3d new_T = result.T_target_source * se3_exp(delta);
        double new_e = reduction.error(target, source, new_T, factors);
        general_factor.update_error(target, source, new_T, &e);

        if (verbose) {
          std::cout << "iter=" << i << " inner=" << j << " e=" << e << " new_e=" << new_e << " lambda=" << lambda << " dt=" << delta.tail<3>().norm()
                    << " dr=" << delta.head<3>().norm() << std::endl;
        }

        if (new_e <= e) {
          // Error decreased, decrease lambda
          result.converged = criteria.converged(delta);
          result.T_target_source = new_T;
          lambda /= lambda_factor;
          success = true;
          e = new_e;

          break;
        } else {
          // Failed to decrease error, increase lambda
          lambda *= lambda_factor;
        }
      }

      result.iterations = i;
      result.H = H;
      result.b = b;
      result.error = e;

      if (!success) {
        break;
      }
    }

    result.num_inliers = std::count_if(factors.begin(), factors.end(), [](const auto& factor) { return factor.inlier(); });

    return result;
  }

  bool verbose;              ///< If true, print debug messages
  int max_iterations;        ///< Max number of optimization iterations
  int max_inner_iterations;  ///< Max  number of inner iterations (lambda-trial)
  double init_lambda;        ///< Initial lambda (damping factor)
  double lambda_factor;      ///< Lambda increase factor
};

struct RobustLevenbergMarquardtOptimizer {
  RobustLevenbergMarquardtOptimizer()
      : verbose(false),
        max_iterations(20),
        max_inner_iterations(20),
        tau(1e-2) {}

  template <
      typename TargetPointCloud,
      typename SourcePointCloud,
      typename TargetTree,
      typename CorrespondenceRejector,
      typename TerminationCriteria,
      typename Reduction,
      typename Factor,
      typename GeneralFactor>
  RegistrationResult optimize(const TargetPointCloud& target,
                              const SourcePointCloud& source,
                              const TargetTree& target_tree,
                              const CorrespondenceRejector& rejector,
                              const TerminationCriteria& criteria,
                              Reduction& reduction,
                              const Eigen::Isometry3d& init_T,
                              std::vector<Factor>& factors,
                              GeneralFactor& general_factor) const {
    if (verbose) {
      std::cout << "--- LM optimization ---" << std::endl;
    }

    double v = 2.0;
    double lambda = 0.0;

    RegistrationResult result(init_T);
    // SolutionRemappingHandler remap_handler;
    double e = 0.0;

    for (int i = 0; i < max_iterations && !result.converged; ++i) {
      // Linearize system
      auto [H, b, e_init] = reduction.linearize(target, source, target_tree, rejector, result.T_target_source, factors);
      general_factor.update_linearized_system(target, source, target_tree, result.T_target_source, &H, &b, &e_init);
      e = e_init;
      lambda = tau * H.diagonal().maxCoeff();

      // remap_handler.compute(H);

      bool step_accepted = false;

      for (int j = 0; j < max_inner_iterations; ++j) {
        // Apply damping
        Eigen::Matrix<double, 6, 6> H_damped = H + lambda * Eigen::Matrix<double, 6, 6>::Identity();

        // Adaptive diagonal (Jacobi) preconditioner
        Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Zero();
        for (int k = 0; k < 6; ++k) {
          double d = std::max(H_damped(k, k), 1e-12);
          P(k, k) = 1.0 / std::sqrt(d);
        }

        // Preconditioned system
        Eigen::Matrix<double, 6, 6> A = P * H_damped * P;
        Eigen::Matrix<double, 6, 1> rhs = -P * b;

        // Solve A * delta_hat = rhs
        Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix<double, 6, 1> delta_hat;
        {
          const double tol = 1e-12;
          Eigen::Matrix<double, 6, 6> S_inv = Eigen::Matrix<double, 6, 6>::Zero();
          const auto& sing = svd.singularValues();
          for (int k = 0; k < 6; ++k)
            if (sing[k] > tol) S_inv(k, k) = 1.0 / sing[k];

          delta_hat = svd.matrixV() * S_inv * svd.matrixU().transpose() * rhs;
        }

        Eigen::Matrix<double, 6, 1> delta = P * delta_hat;
        if (!delta.allFinite()) {
          if (verbose) std::cerr << "[LM] Invalid delta (NaNs)\n";
          break;
        }
        // remap_handler.apply(delta);

        const Eigen::Isometry3d new_T = result.T_target_source * se3_exp(delta);
        double new_e = reduction.error(target, source, new_T, factors);
        general_factor.update_error(target, source, new_T, &new_e);

        // Gain ratio
        double expected = -delta.dot(b) - 0.5 * delta.dot(H * delta);
        double rho = (e - new_e) / std::max(expected, 1e-15);

        if (verbose) {
          double cond_H = svd.singularValues()(0) / std::max(svd.singularValues()(5), 1e-12);
          std::cout << "iter=" << i
                    << " inner=" << j
                    << " e=" << e
                    << " new_e=" << new_e
                    << " ρ=" << rho
                    << " λ=" << lambda
                    << " cond=" << cond_H
                    << " dt=" << delta.tail<3>().norm()
                    << " dr=" << delta.head<3>().norm()
                    << std::endl;
        }

        if (rho > 0 && std::isfinite(new_e) && std::isfinite(delta.norm())) {
          result.T_target_source = new_T;
          e = new_e;
          lambda = lambda * std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
          lambda = std::max(lambda, 1e-9); // Avoid too small lambda
          step_accepted = true;
          result.converged = criteria.converged(delta) || std::abs(expected) < 1e-12;
          // Reset v for future additive updates
          v = 2.0;
          break;
        } else {
          lambda = lambda + v;    // Additive increase (Nielsen’s method)
          v = v * 2.0;            // Exponential growth to escape poor regions
        }

      }

      result.iterations = i + step_accepted;
      result.H = H;
      result.b = b;
      result.error = e;

      if (!step_accepted) {
        if (verbose) std::cerr << "[LM] Step rejected, terminating\n";
        break;
      }
    }

    result.num_inliers = static_cast<int>(
        std::count_if(factors.begin(), factors.end(),
                      [](const auto& f) { return f.inlier(); }));

    return result;
  }

  bool verbose;
  int max_iterations;
  int max_inner_iterations;
  double  tau;
};

struct HouseholderSolver {
  HouseholderSolver()
      : verbose(false),
        max_iterations(30),
        step_shrink(0.5),    // back-tracking factor
        min_step_scale(1e-3) // give up if step is scaled below this
  {}

  template <
      typename TargetPointCloud,
      typename SourcePointCloud,
      typename TargetTree,
      typename CorrespondenceRejector,
      typename TerminationCriteria,
      typename Reduction,
      typename Factor,
      typename GeneralFactor>
  RegistrationResult optimize(const TargetPointCloud& target,
                              const SourcePointCloud& source,
                              const TargetTree& target_tree,
                              const CorrespondenceRejector& rejector,
                              const TerminationCriteria& criteria,
                              Reduction& reduction,
                              const Eigen::Isometry3d& init_T,
                              std::vector<Factor>& factors,
                              GeneralFactor& general_factor) const {
    if (verbose) std::cout << "--- Householder Gauss-Newton optimisation ---\n";

    RegistrationResult result(init_T);
    double e = std::numeric_limits<double>::max();

    for (int i = 0; i < max_iterations && !result.converged; ++i) {
      /* ---------- build normal-equation ---------- */
      auto [H, b, e_init] = reduction.linearize(
          target, source, target_tree, rejector, result.T_target_source, factors);
      general_factor.update_linearized_system(
          target, source, target_tree, result.T_target_source, &H, &b, &e_init);

      /* ---------- solve H δ = –b ---------- */
      Eigen::Matrix<double, 6, 1> delta =
          Eigen::HouseholderQR<Eigen::Matrix<double, 6, 6>>(H).solve(-b);
      if (!delta.allFinite()) throw std::runtime_error("[HouseholderSolver] NaNs in δ");

      /* ---------- back-tracking line-search ---------- */
      double scale = 1.0;
      Eigen::Isometry3d new_T;
      double new_e = e_init;
      bool accepted = false;
      while (scale >= min_step_scale) {
        new_T = result.T_target_source * se3_exp(scale * delta);
        new_e = reduction.error(target, source, new_T, factors);
        general_factor.update_error(target, source, new_T, &new_e);

        if (new_e < e_init) {          // improvement
          accepted = true;
          break;
        }
        scale *= step_shrink;          // shrink step and retry
      }
      if (!accepted) {
        if (verbose) std::cerr << "[HouseholderSolver] no descent, terminating\n";
        break;                         // leave optimisation
      }

      /* ---------- commit step ---------- */
      result.T_target_source = new_T;
      e = new_e;
      result.error = new_e;
      result.H = H;
      result.b = b;
      result.iterations = i + 1;

      if (verbose) {
        std::cout << "iter=" << i
                  << "  e=" << e_init
                  << "  new_e=" << new_e
                  << "  scale=" << scale
                  << "  |δr|=" << delta.head<3>().norm()
                  << "  |δt|=" << delta.tail<3>().norm()
                  << '\n';
      }

      result.converged = criteria.converged(scale * delta);
    }

    result.num_inliers = static_cast<int>(std::count_if(
        factors.begin(), factors.end(), [](const auto& f) { return f.inlier(); }));

    return result;
  }

  bool verbose;
  int  max_iterations;
  double step_shrink;
  double min_step_scale;
};

}  // namespace small_gicp
