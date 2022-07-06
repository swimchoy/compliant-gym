// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef RSG_RAIBOT__COMPLIANT_CONTACT_HPP_
#define RSG_RAIBOT__COMPLIANT_CONTACT_HPP_

#include "contactContainer.hpp"
#include "terrainParameters.hpp"

namespace raisim
{

class CompliantContact {
 private:
  bool active_;

  double m_added_;
  double z_prev_;
  double dz_;
  double dz_prev_;
  double F_p_;
  double F_g_;
  raisim::Vec<3> max_penetration_;

  bool isTraversing_;
  double dist_prev_;
  double dist_max_;
  double z_max_;
  raisim::Vec<3> pos_marker_;

  double rigidity_;
  double F_ema_;

  std::shared_ptr<GM> gm_;

  void accumulateFp(const double &z) {
    F_p_ = 0.;
    double delta_z_tmp = 0.005;
    for (double z_tmp = 0.0; z_tmp < z; z_tmp += delta_z_tmp) {
      double r_tmp = z_tmp > FOOT::r
                     ? FOOT::r
                     : std::sqrt(FOOT::r * FOOT::r - (FOOT::r - z_tmp) * (FOOT::r - z_tmp));
      double R = 8 * r_tmp * FOOT::h / (2 * (2 * r_tmp + FOOT::h));

      double A_flat = 3.14159 * (R * R +
          std::pow(gm_->mu * (z_tmp - 0) / std::tan(gm_->theta), 2) -
          2 * R * gm_->mu * (z_tmp - 0) / std::tan(gm_->theta));
      if (R -  (gm_->mu * z / std::tan(gm_->theta)) < 0) {
        A_flat = 0.;
      }
      double A_cone = (3.14159 * R * R - A_flat) / std::cos(gm_->theta);

      F_p_ += gm_->k_div_A * A_flat * delta_z_tmp;
      F_p_ += gm_->sigma_rft * A_cone * delta_z_tmp;
    }
  }

  double traverseDepth(const double &dist, const double &z, const double &z_max) {
    return dist + FOOT::r / std::sin(Crater::gamma) - Crater::r_c - (
        z_max - z + FOOT::r) / std::tan(Crater::gamma);
  }

  static double norm2d(const raisim::Vec<2> &p1, const raisim::Vec<2> &p2) {
    return std::sqrt(std::pow(p1[0] - p2[0], 2) + std::pow(p1[1] - p2[1], 2));
  }

 public:
  CompliantContact()
  : active_(false),
    dz_prev_(0.0),
    dz_(0.0),
    rigidity_(0.0)
  {
  }

  void activate(raisim::ContactContainer *contact, GM *gm) {
    active_ = true;
    gm_ = std::make_shared<GM>(gm->params());

    m_added_ = 0.;
    z_prev_ = 0.;
    dz_ = 0.;
    F_p_ = 0.;
    F_g_ = 0.;
    max_penetration_.setZero();

    isTraversing_ = false;
    dist_prev_ = 0.;
    dist_max_ = 0.;
    z_max_ = 0.;
    pos_marker_.setZero();

    rigidity_ = 0.;
    F_ema_ = 0.;

    double z = contact->pos[2] < 0. ? -contact->pos[2] : 0.0;
    accumulateFp(z);
  }

  void deactivate() {
    F_g_ = 0.;
    active_ = false;
  }

  bool isActive() {
    return active_;
  }

  void update(raisim::World *world, raisim::ContactContainer *contact) {

    double z = contact->pos[2] < 0. ? -contact->pos[2] : 0.0;
    dz_ = -contact->vimp_prev[2];
    double ddz = (dz_ - dz_prev_) / world->getTimeStep();
    double delta_z = z - z_prev_;

    rigidity_ += 0.025 * static_cast<double>(dz_ * dz_prev_ < 0);
    rigidity_ = std::min(rigidity_, 1.0);

    dz_prev_ = dz_;
    z_prev_ = z;

    if (max_penetration_[2] < z) {
      max_penetration_ = {contact->pos[0], contact->pos[1], z};
      z_max_ = z;
    } else {
      double dist = norm2d(contact->pos.head<2>(), max_penetration_.head<2>());
      if (dist > Crater::r_c) {
        if (dist > dist_prev_) {
          if (dist_prev_ < 1e-8) {
            dist_prev_ = Crater::r_c;
          }
          max_penetration_[2] -= std::tan(Crater::gamma) * (dist - dist_prev_);
          dist_prev_ = dist;
          if (max_penetration_[2] < 0.) {
            max_penetration_[2] = 0.;
          }
        }
        if (traverseDepth(dist, z, z_max_) > 0) {
          if (!isTraversing_) {
            pos_marker_ = {max_penetration_[0], max_penetration_[1], z_max_};
            isTraversing_ = true;
          }
          m_added_ = 0.;
          accumulateFp(z);
          rigidity_ = 0.;
          dist_prev_ = 0.;
        }
      }
    }

    double r_tmp = z > FOOT::r
                   ? FOOT::r
                   : std::sqrt(FOOT::r * FOOT::r - (FOOT::r - z) * (FOOT::r - z));
    double R = 8 * r_tmp * FOOT::h / (2 * (2 * r_tmp + FOOT::h)); // hydraulic radius 4A / P

    double A_flat = 3.14159 * (R * R +
        std::pow(gm_->mu * (z - 0) / std::tan(gm_->theta), 2) -
        2 * R * gm_->mu * (z - 0) / std::tan(gm_->theta));
    if (R -  (gm_->mu * z / std::tan(gm_->theta)) < 0) {
      A_flat = 0.;
    }
    double A_cone = (3.14159 * R * R - A_flat) / std::cos(gm_->theta);

    m_added_ += gm_->c_g * gm_->phi * gm_->rho * gm_->mu * A_flat * delta_z;

    F_p_ += gm_->k_div_A * A_flat * delta_z;
    F_p_ += gm_->sigma_rft * A_cone * delta_z;

    if (dz_ > 0 && max_penetration_[2] - 1e-4 <= z)
    {
      double m_added_dot = gm_->c_g * gm_->phi * gm_->rho * gm_->mu * A_flat * dz_;
      F_g_ = F_p_ + gm_->b * m_added_dot * dz_ + m_added_ * ddz;
    }
    else
    {
      F_g_ = 0.;
    }

    double alpha = 1.0 - 0.8 * rigidity_;
    F_ema_ = alpha * F_g_ + (1- alpha) * F_ema_;
  }

  void updateFromUnContact(double v_z) {
    F_g_ = 0.;
    dz_ = -v_z;
    dz_prev_ = dz_;
  }

  double getFilteredNormalImp(double dt) {
    return F_ema_ * dt;
  }

  double getNormalForTangential(double dt) {
    return F_p_ * dt;
  }

  double getMaxPenetrationDepth() {
    return max_penetration_[2];
  }

  bool isTraversing() {
    return isTraversing_;
  }

  bool getTraverseResistive(raisim::ContactContainer *contact, raisim::Vec<3> &F_r, raisim::Vec<3> &pos_r) {
    if (isTraversing_) {
      double dist = norm2d(contact->pos.head<2>(), pos_marker_.head<2>());
      double z = contact->pos[2] < 0. ? -contact->pos[2] : 0.0;
      double t_depth = traverseDepth(dist, z, pos_marker_[2]);
      if (t_depth > 0 && dist_max_ < dist && z > FOOT::r) {
        double F_r_elastic = gm_->k_t * (t_depth + 0.5 * (z - FOOT::r));

        F_r = {-F_r_elastic * (contact->pos[0] - pos_marker_[0]) / dist,
               -F_r_elastic * (contact->pos[1] - pos_marker_[1]) / dist,
               0};

        if ((contact->vimp_prev[0] < 0) == (contact->pos[0] - pos_marker_[0] < 0))
          F_r[0] -= gm_->b_t * contact->vimp_prev[0];
        if ((contact->vimp_prev[1] < 0) == (contact->pos[1] - pos_marker_[1] < 0))
          F_r[1] -= gm_->b_t * contact->vimp_prev[1];

        pos_r = contact->pos;
        pos_r[0] += std::copysign(FOOT::r,  pos_r[0] - pos_marker_[0]);
        pos_r[1] += std::copysign(FOOT::r,  pos_r[0] - pos_marker_[0]);
        pos_r[2] += FOOT::r;

        dist_max_ = dist;
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

};

} // namespace raisim

#endif  //RSG_RAIBOT__COMPLIANT_CONTACT_HPP_