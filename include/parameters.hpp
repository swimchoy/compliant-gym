// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef _COMPLIANT_GYM__TERRAIN_PARAMETER_HPP_
#define _COMPLIANT_GYM__TERRAIN_PARAMETER_HPP_

#include "contactContainer.hpp"

namespace raisim
{

struct FOOT {
  static constexpr double r = 0.0295;
  static constexpr double h = 0.025;
};

struct Crater {
  static constexpr double r_c = 0.05;
  static constexpr double gamma = 3.14159 / 4.;
};

struct GM {

  explicit GM (Eigen::VectorXd params) {
    nu = params[0];
    theta = params[1];
    c_g = params[2];
    phi = params[3];
    rho = params[4];
    sigma_flat = params[5];
    c_d = params[6];
    sigma_cone = params[7];
    k_h = params[8];
    b_h = params[9];
  }

  GM () {}

  double nu = 2.0;
  double theta = 3.14159 / 3.;
  double c_g = 2.7;
  double phi = 0.57;
  double rho = 1000;
  double sigma_flat = 1.0e6;
  double c_d = 17.2;
  double sigma_cone = 0.5e6;
  double k_h = 160.0;
  double b_h = 1.0;

  Eigen::VectorXd params() {
    Eigen::VectorXd p(10);
    p << nu, theta, c_g, phi, rho, sigma_flat, c_d, sigma_cone, k_h, b_h;
    return p;
  }
};

} // namespace raisim

#endif  //_COMPLIANT_GYM__TERRAIN_PARAMETER_HPP_