//
// Created by suyoung on 1/25/22.
//

#ifndef _COMPLIANT_GYM_TESTER_SRC_RAIBOT_COMPLIANT_CONTROLLER_HPP_
#define _COMPLIANT_GYM_TESTER_SRC_RAIBOT_COMPLIANT_CONTROLLER_HPP_

#include <filesystem>
#include "BasicEigenTypes.hpp"
#include "raibotController.hpp"
#include "neuralNet.hpp"

namespace controller {

class raibotCompliantController {

 public:
  raibotCompliantController();
  bool create(raisim::World *world);
  void setTimeConfig(double control_dt, double simulation_dt);
  bool reset(raisim::World *world);
  Eigen::VectorXf obsScalingAndGetAction();
  bool advance(raisim::World *world);
  void setCommand(const Eigen::Ref<raisim::EigenVec>& command);
  Eigen::VectorXf getEstimation();

 private:
  raisim::raibotController raibotController_;
  Eigen::VectorXf obs_;
  Eigen::VectorXf estimation_;
  Eigen::VectorXf actor_input_;

  Eigen::VectorXf obsMean_;
  Eigen::VectorXf obsVariance_;
  Eigen::VectorXf eoutMean_;
  Eigen::VectorXf eoutVariance_;
  raisim::nn::LSTM<float, 30, 64> encoder_;
  raisim::nn::Linear<float, 75, 12, raisim::nn::ActivationType::leaky_relu> actor_;
  raisim::nn::Linear<float, 64, 8, raisim::nn::ActivationType::leaky_relu> estimator_;

  int clk_ = 0;
  double control_dt_;
  double communication_dt_;
};

} // namespace raisim

#endif //_COMPLIANT_GYM_TESTER_SRC_RAIBOT_COMPLIANT_CONTROLLER_HPP_
