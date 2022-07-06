// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef RSG_RAIBOT__CONTACT_CONTAINER_HPP_
#define RSG_RAIBOT__CONTACT_CONTAINER_HPP_

namespace raisim
{

struct ContactContainer {
  bool isRobotObjA;
  double mu;
  raisim::Vec<3> imp;
  raisim::Vec<3> vimp;
  raisim::Vec<3> vimp_prev;
  raisim::Vec<3> pos;
  raisim::Mat<3, 3> rot;
  raisim::Mat<3, 3> MappInv_W;
  raisim::SparseJacobian sparse_jaco;
  raisim::Vec<3> dtRTJMinvTauMinusB;
  raisim::MatDyn MinvJTR;
  raisim::SparseJacobian RTJ;
};

} // namespace raisim

#endif  //RSG_RAIBOT__CONTACT_CONTAINER_HPP_