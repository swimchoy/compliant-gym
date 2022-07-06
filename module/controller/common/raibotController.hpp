// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef RSG_RAIBOT__RAIBOTCONTROLLER_HPP_
#define RSG_RAIBOT__RAIBOTCONTROLLER_HPP_

#include <set>
#include "raisim/World.hpp"

namespace raisim {

class raibotController {

 public:

  bool create(raisim::World *world) {
    raibot_ = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// get robot data
    gcDim_ = raibot_->getGeneralizedCoordinateDim();
    gvDim_ = raibot_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gc_init_.setZero(gcDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of raibot
    gc_init_ << 0, 0, 0.4725, 1, 0.0, 0.0, 0.0,
                0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672, 0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672;
    raibot_->setState(gc_init_, gv_init_);

    /// set pd gains
    jointPGain_.setZero(gvDim_);
    jointDGain_.setZero(gvDim_);
    jointPGain_.tail(nJoints_).setConstant(50.0);
    jointDGain_.tail(nJoints_).setConstant(0.5);
    raibot_->setPdGains(jointPGain_, jointDGain_);
    raibot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// vector dimensions
    obDim_ = 33;
    estDim_ = 8;
    actionDim_ = nJoints_;

    historyLength_ = 20;
    history_.resize(historyLength_);
    historyFloat_.setZero(historyLength_ * obDim_);

    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    command_.setZero();

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    updateObservation(world);

    for (int i = 0; i < historyLength_; ++i) {
      history_[i].setZero(obDim_);
      history_[i] = obDouble_.cast<float>();
    }

    return true;
  }

  bool init(raisim::World *world) { return true; }

  bool reset(raisim::World *world) {
    command_ << 0., 0., 0.;
    updateObservation(world);
    for (int i = 0; i < historyLength_; ++i) {
      history_[i].setZero(obDim_);
      history_[i] = obDouble_.cast<float>();
    }
    return true;
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec>& action) {

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    raibot_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  void updateObservation(raisim::World *world) {
    raibot_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot_);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);

    obDouble_ << command_, /// command 3
        rot_.e().row(2).transpose(), /// body orientation: z-axis 3
        bodyAngularVel_, /// body angular velocity 3
        gc_.tail(12), /// joint angles 12
        gv_.tail(12); /// joint velocity 12
  }

  void updateHistory(Eigen::Ref<EigenVec> ob) {
    std::rotate(history_.begin(), history_.begin()+1, history_.end());
    history_[historyLength_ - 1] = ob.head(obDim_);

    for (int i = 0; i < historyLength_; ++i)
      historyFloat_.segment(i * obDim_, obDim_) = history_[i];
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    command_ = command.cast<double>();

    // project to centrifugal accel. vxy * wz = 0.3 * g
    if (std::abs(command_(2)) - 2.943 / (command_.head(2).norm() + 1e-8) > 0) {
      command_(2) = std::copysign(2.943 / (command_.head(2).norm() + 1e-8), command_(2));
    }
  }

  void setEstDim(int estDim) { estDim_ = estDim; }

  Eigen::Vector3d getCommand() { return command_; }

  const Eigen::VectorXd& getObservation() { return obDouble_; }

  const Eigen::VectorXf& getHistory() { return historyFloat_; }

  int getObDim() const { return obDim_; }

  int getEstDim() const { return estDim_; }

  int getActionDim() const { return actionDim_; }

  int getHistLength() const { return historyLength_; }

  Eigen::VectorXd getJointPGain() const { return jointPGain_; }

  Eigen::VectorXd getJointDGain() const { return jointDGain_; }

  Eigen::VectorXd getJointPTarget() const { return pTarget12_; }

  void getInitState(Eigen::VectorXd &gc, Eigen::VectorXd &gv) {
    gc.resize(gcDim_);
    gv.resize(gvDim_);
    gc << gc_init_;
    gv << gv_init_;
  }

private:
  raisim::ArticulatedSystem *raibot_;

  int gcDim_;
  int gvDim_;
  int nJoints_;
  Eigen::VectorXd gc_init_;
  Eigen::VectorXd gv_init_;
  Eigen::VectorXd jointPGain_;
  Eigen::VectorXd jointDGain_;

  Eigen::VectorXd gc_;
  Eigen::VectorXd gv_;
  raisim::Mat<3,3> rot_;
  Eigen::Vector3d bodyAngularVel_;

  int obDim_;
  int estDim_;
  int actionDim_;
  Eigen::Vector3d command_;
  Eigen::VectorXd obDouble_;

  int historyLength_;
  Eigen::VectorXf historyFloat_;
  std::vector<Eigen::VectorXf> history_;

  Eigen::VectorXd pTarget_;
  Eigen::VectorXd pTarget12_;
  Eigen::VectorXd vTarget_;
  Eigen::VectorXd actionMean_;
  Eigen::VectorXd actionStd_;
};

} // namespace raisim

#endif    // RSG_RAIBOT__RAIBOTCONTROLLER_HPP_