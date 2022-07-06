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
    mu = params[0];
    theta = params[1];
    c_g = params[2];
    phi = params[3];
    rho = params[4];
    k_div_A = params[5];
    b = params[6];
    sigma_rft = params[7];
    k_t = params[8];
    b_t = params[9];
  }

  GM () {}

  double mu = 2.0;                      // recruitment rate
  double theta = 3.14159 / 3.;          // shear band angle (60 deg)
  double c_g = 2.7;            // surrounding mass scaling factor
  double phi = 0.57;            // volume fraction (loose-packed)
  double rho = 1000;            // grain density (1000 kg/m3 for poppy seed)
  double k_div_A = 1.0e6;        // penetration stiffness divided by foot area
  double b = 17.2;              // inertial drag scaling factor
  double sigma_rft = 0.15e6;      // depth dependent resistive stress
  double k_t = 160.0;            // custom: traverse resistive stiffness
  double b_t = 1.0;            // custom: traverse resistive damping

  Eigen::VectorXd params() {
    Eigen::VectorXd p(10);
    p << mu, theta, c_g, phi, rho, k_div_A, b, sigma_rft, k_t, b_t;
    return p;
  }
};

} // namespace raisim

#endif  //_COMPLIANT_GYM__TERRAIN_PARAMETER_HPP_