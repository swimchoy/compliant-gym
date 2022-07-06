//
// Created by suyoung on 6/17/22.
//

#ifndef _COMPLIANT_GYM__ADDED_MASS_HPP_
#define _COMPLIANT_GYM__ADDED_MASS_HPP_

#include "compliantContact.hpp"
#include "terrainParameters.hpp"

namespace cylinderContact
{

class addedMass {
 private:
  raisim::CompliantContact oursContact_;
  raisim::ContactContainer contact_;

  static raisim::Vec<2> prox_t (const Eigen::Vector2d & in, const double &lam_z, const double &mu) {
    if (in.norm() <= mu * lam_z) {
      return in;
    } else {
      return ((mu * lam_z) / in.norm()) * in;
    }
  }

 public:
  void advance(raisim::World* world, raisim::Cylinder* cyl, raisim::GM *gm) {
    contact_.pos = cyl->getPosition();
    contact_.pos[2] -= raisim::FOOT::r;
    contact_.vimp_prev = cyl->getLinearVelocity();

    bool isActive = oursContact_.isActive();

    if (contact_.pos[2] <= 0. && !isActive) {
      oursContact_.activate(&contact_, gm);
      oursContact_.update(world, &contact_);
    }

    else if (contact_.pos[2] > 0. && isActive) {
      oursContact_.deactivate();
    }

    else if (contact_.pos[2] <= 0. && isActive) {
      oursContact_.update(world, &contact_);
    }

    auto simulation_dt = world->getTimeStep();
    double mu = world->getMaterialPairProperties("default", "default").c_f;
    double c_t = 0.5 / cyl->getMass();

    contact_.imp = {0., 0., oursContact_.getFilteredNormalImp(simulation_dt)};
    double err = 1;
    while (err > 1e-5) {
      err = 0;
      contact_.vimp = contact_.vimp_prev + contact_.imp / cyl->getMass();
      raisim::Vec<2> next_imp_t = prox_t(contact_.imp.e().head(2) - contact_.vimp.e().head(2) * c_t,
                                         oursContact_.getNormalForTangential(simulation_dt),
//                                         contact_.imp[2],
                                         mu);
      err += (contact_.imp.e().head(2) - next_imp_t.e()).norm();
      contact_.imp.head<2>() = next_imp_t;
    }

    if (oursContact_.isActive()) {
      cyl->setExternalForce(cyl->getIndexInWorld(), contact_.imp / simulation_dt);
    }

  }

  bool isActive() {
    return oursContact_.isActive();
  }

};

} // namespace raisim

#endif //_COMPLIANT_GYM__ADDED_MASS_HPP_