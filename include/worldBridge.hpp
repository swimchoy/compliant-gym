// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef RSG_RAIBOT__WORLD_BRIDGE_HPP_
#define RSG_RAIBOT__WORLD_BRIDGE_HPP_

#include <bitset>

#include "contactContainer.hpp"
#include "compliantContact.hpp"

namespace raisim
{

class worldBridge {
 public:
  void create(raisim::World *world) {
    robot_ = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    gc_.setZero(robot_->getGeneralizedCoordinateDim());
    gv_.setZero(robot_->getDOF());
    gf_.setZero(robot_->getDOF());

    M_inv_.resize(robot_->getDOF(), robot_->getDOF());
    b_.resize(robot_->getDOF());

    PGain_.setZero(robot_->getDOF());
    DGain_.setZero(robot_->getDOF());

    tauMinusB_.setZero(robot_->getDOF());
    MinvTauMinusB_.setZero(robot_->getDOF());

    footIndices_.push_back(robot_->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(robot_->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(robot_->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(robot_->getBodyIdx("RH_SHANK"));

    Contacts_.resize(footIndices_.size());
    compliantContacts_.resize(footIndices_.size());

    for (auto &contact: Contacts_) {
      contact.MinvJTR.setZero(robot_->getDOF(), 3);
    }

    std::vector<std::string> boxNames = {"LF_box", "RF_box", "LH_box", "RH_box"};
    shankColObjNames_ = {"LF_SHANK/0", "RF_SHANK/0", "LH_SHANK/0", "RH_SHANK/0"};
    footColObjNames_ = {"LF_FOOT/0", "RF_FOOT/0", "LH_FOOT/0", "RH_FOOT/0"};

    for (int i = 0; i < boxNames.size(); ++i) {
      boxes_.push_back(world->addBox(0.25, 0.25, 0.05, 1.0, "default",
                                    raisim::COLLISION(i+2), raisim::COLLISION(i+2)));
      boxes_.back()->setName(boxNames[i]);
      robot_->getCollisionBody(shankColObjNames_[i]).setCollisionGroup(raisim::COLLISION(i+2));
      robot_->getCollisionBody(shankColObjNames_[i]).setCollisionMask(raisim::COLLISION(i+2) | raisim::COLLISION(63));
      robot_->getCollisionBody(footColObjNames_[i]).setCollisionGroup(raisim::COLLISION(i+2));
      robot_->getCollisionBody(footColObjNames_[i]).setCollisionMask(raisim::COLLISION(i+2) | raisim::COLLISION(63));
    }

    footPosition_.resize(footIndices_.size());
    footCollisionOffset_.clear();
    for (int i = 0; i < footIndices_.size(); ++i) {
      footCollisionOffset_.push_back({0., 0., -0.24});
    }
  }

  void setTerrainParameters(GM gm) {
    gm_ = gm;
  }

  void reset() {

    for (auto &box: boxes_) {
      box->setPosition(0., 0., -0.025);
      box->setVelocity(0.,0.,0.,0.,0.,0.);
    }

    robot_->getPdGains(PGain_, DGain_);
    clk_ = 0;
//    compliantContacts_.clear();
    contactEligibilityTrace_.setZero();
    contactStatus_.resize(4);
  }

  void setFootCollisionPosOffset(const std::vector<raisim::Vec<3>> &footOffset) {
    if (footCollisionOffset_.size() != footOffset.size()) {
      footCollisionOffset_.resize(footOffset.size());
    }
    std::copy(footOffset.begin(), footOffset.end(), footCollisionOffset_.begin());
  }

  void updateBoxPositions(raisim::World *world) {

    raisim::Mat<3, 3> shankOrientation{};
    raisim::Vec<3> shankPosition;

    for (int i = 0; i < footIndices_.size(); ++i) {
      robot_->getBodyPose(footIndices_[i], shankOrientation, shankPosition);
      footPosition_[i] = shankPosition + shankOrientation * footCollisionOffset_[i];
      footPosition_[i][2] -= 0.0295;
      double box_z_position = footPosition_[i][2] > 0.0 ? -0.025 : footPosition_[i][2] - 0.025;
      boxes_[i]->setPosition(footPosition_[i][0], footPosition_[i][1], box_z_position);
      boxes_[i]->setOrientation(1., 0., 0., 0.);
    }

  }

  void integrate(raisim::World *world) {

    updateBoxPositions(world);

    world->integrate1();

    updateRobotStatus(world, robot_, 10);

    updateContactInfo(world, robot_);

    updateSoftContactForce(world, robot_);

    solveSoftContact(world, robot_);

    setTraverseResistiveForce(world, robot_);     // optional

    maskOnFootCollision(15);

    world->integrate();

    maskOffFootCollision();

    updateContactEligibility();     // optional

  }

  void updateRobotStatus (raisim::World *world, raisim::ArticulatedSystem *robot, size_t max_clk) {

    gc_ = robot->getGeneralizedCoordinate();
    gv_ = robot->getGeneralizedVelocity();

    if (clk_ == 0) {
      robot->getMassMatrix();
      M_inv_ = robot->getInverseMassMatrix();
    }
    gf_.setZero();
    b_ = robot->getNonlinearities(world->getGravity());

    Eigen::VectorXd pTarget(robot->getGeneralizedCoordinateDim());
    Eigen::VectorXd dTarget(robot->getDOF());
    robot->getPdTarget(pTarget, dTarget);

    for (int i = 0; i < 12; ++i) {
      gf_[i + 6] = PGain_.tail(12)[i] * (pTarget.tail(12)[i] - gc_.e().tail(12)[i]) +
          DGain_.tail(12)[i] * (dTarget.tail(12)[i] - gv_.e().tail(12)[i]) +
          robot->getFeedForwardGeneralizedForce()[i];
    }

    raisim::vecsub(gf_, b_, tauMinusB_);
    raisim::matvecmul(M_inv_, tauMinusB_, MinvTauMinusB_);

    clk_ = ++clk_ >= max_clk ? 0 : clk_;
  }

  void updateContactInfo (raisim::World *world, raisim::ArticulatedSystem *robot) {
    double mu = world->getMaterialPairProperties("default", "default").c_f;

    inContact_.clear();
    isRigidContact_.reset();

    for (auto& contact: robot->getContacts()) {
      if (contact.skip()) continue;
      size_t idx = std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) - footIndices_.begin();
      if (idx == footIndices_.size()) {
        continue;
      } else {
        isRigidContact_[idx] = (contact.getPairObjectBodyType() == raisim::BodyType::STATIC) || isRigidContact_[idx];
        if (!inContact_.count(idx))
        {
          inContact_.insert(idx);
          Contacts_[idx].pos = contact.getPosition();
          if (Contacts_[idx].pos[2] > 0) continue;
          Contacts_[idx].rot.setIdentity();     // contact.getContactFrame();
          Contacts_[idx].isRobotObjA = true;    // contact.isObjectA();

          raisim::Vec<3> vel_w;
          robot->getSparseJacobian(footIndices_[idx], Contacts_[idx].pos, Contacts_[idx].sparse_jaco);
          robot->getVelocity(Contacts_[idx].sparse_jaco, vel_w);

          raisim::matTransposevecmul(Contacts_[idx].rot, vel_w, Contacts_[idx].vimp_prev);

          raisim::MatDyn MinvJT(robot->getDOF(), 3);
          matSparseJacoTransposeMul(M_inv_, Contacts_[idx].sparse_jaco, MinvJT);
          sparseJacoMatMul(Contacts_[idx].sparse_jaco, MinvJT, Contacts_[idx].MappInv_W); // rotMat is identity.
          Contacts_[idx].imp = {0., 0., 0.};
          Contacts_[idx].mu = mu;

          raisim::Vec<3> tmpVec3;
          raisim::matmul(Contacts_[idx].sparse_jaco, MinvTauMinusB_, tmpVec3);
          raisim::matTransposevecmul(Contacts_[idx].rot, tmpVec3, Contacts_[idx].dtRTJMinvTauMinusB);
          raisim::vecScalarMul(world->getTimeStep(), Contacts_[idx].dtRTJMinvTauMinusB);

          raisim::matmul(MinvJT, Contacts_[idx].rot, Contacts_[idx].MinvJTR);
          matTransposeSparseJaco(Contacts_[idx].rot, Contacts_[idx].sparse_jaco, Contacts_[idx].RTJ);
        }
      }
    }

  }

  void updateSoftContactForce (raisim::World *world, raisim::ArticulatedSystem *robot) {

    for (size_t idx = 0; idx < footIndices_.size(); ++idx) {

      if (inContact_.count(idx)) {

        bool isActive = compliantContacts_[idx].isActive();

        if (footPosition_[idx][2] <= 0. && !isActive) {
          compliantContacts_[idx].activate(&Contacts_[idx], &gm_);
          compliantContacts_[idx].update(world, &Contacts_[idx]);
        }

        else if (footPosition_[idx][2] > 0. && isActive) {
          compliantContacts_[idx].deactivate();
        }

        else if (footPosition_[idx][2] <= 0. && isActive) {
          compliantContacts_[idx].update(world, &Contacts_[idx]);
        }

      } else {
        raisim::Vec<3> footVel;
        raisim::SparseJacobian j;

        robot->getSparseJacobian(footIndices_[idx], footPosition_[idx], j);
        robot->getVelocity(j, footVel);

        compliantContacts_[idx].updateFromUnContact(footVel[2]);
      }

    }
  }

  void solveSoftContact (raisim::World* world, raisim::ArticulatedSystem* robot) {
    double c_z;
    double c_t;
    double alpha = 0.5;
    double totalErr = 1.0;

    if (!inContact_.empty()) {

      int loopCount = 0;

      while (totalErr > 1e-5) {

        totalErr = 0.;

        for (auto idx: inContact_) {
          auto contact = &Contacts_[idx];

          raisim::matvecmul(contact->MappInv_W, contact->imp, contact->vimp);
          raisim::vecadd(contact->vimp_prev, contact->vimp);

          if (contact->isRobotObjA) {
            raisim::vecadd(contact->dtRTJMinvTauMinusB, contact->vimp);
          } else {
            raisim::vecsub(contact->dtRTJMinvTauMinusB, contact->vimp);
          }

          for (auto j: inContact_) {
            if (j == idx) continue;
            auto contact_j = &Contacts_[j];
            if (contact->isRobotObjA && contact_j->isRobotObjA) {

              raisim::VecDyn MinvJTRimp(robot->getDOF());
              matvecmul(contact_j->MinvJTR, contact_j->imp, MinvJTRimp);
              sparseJacoVecMulThenAdd(contact->RTJ, MinvJTRimp, contact->vimp);

            } else if (contact->isRobotObjA && !contact_j->isRobotObjA) {
//              contact->vimp -= contact->rot.e().transpose() * contact->jaco * M_inv_.e() * contact_j->jaco.transpose() * contact_j->rot * contact_j->imp;
            } else if (!contact->isRobotObjA && contact_j->isRobotObjA) {
//              contact->vimp -= contact->rot.e().transpose() * contact->jaco * Zero * contact_j->jaco.transpose() * contact_j->rot * contact_j->imp;
            } else if (!contact->isRobotObjA && !contact_j->isRobotObjA) {
//              contact->vimp += contact->rot.e().transpose() * contact->jaco * Zero * contact_j->jaco.transpose() * contact_j->rot * contact_j->imp;
            }
          }

          c_z = alpha / (contact->MappInv_W)(2, 2);
          c_t = alpha / std::max(contact->MappInv_W(0, 0), contact->MappInv_W(1, 1));

          double next_imp_z;
          raisim::Vec<2> next_imp_t;
          if (isRigidContact_[idx]) {
            next_imp_z = prox_z(contact->imp[2] - contact->vimp[2] * c_z);
            next_imp_t = prox_t(contact->imp.e().head(2) - contact->vimp.e().head(2) * c_t, next_imp_z, contact->mu);
          } else {
            next_imp_z = compliantContacts_[idx].isActive()
                ? prox_z(compliantContacts_[idx].getFilteredNormalImp(world->getTimeStep()))
                : 0.0;
            next_imp_t = prox_t(contact->imp.e().head(2) - contact->vimp.e().head(2) * c_t,
                                compliantContacts_[idx].isActive()
                                ? prox_z(compliantContacts_[idx].getNormalForTangential(world->getTimeStep()))
                                : 0.0,
                                contact->mu);
          }
//          raisim::Vec<2> next_imp_t = prox_t(contact->imp.e().head(2) - contact->vimp.e().head(2) * c_t, next_imp_z, contact->mu);

          totalErr += std::abs(contact->imp[2] - next_imp_z);
          totalErr += (contact->imp.e().head(2) - next_imp_t.e()).norm();
          contact->imp = {next_imp_t[0], next_imp_t[1], next_imp_z};
        }

        if (++loopCount > 150) break;
      }
//      RSWARN_IF(loopCount > 150, "TimeOutWarning: contact solution might not be correct")

      for (auto idx: inContact_) {
        robot->setExternalForce(footIndices_[idx],
                                raisim::ArticulatedSystem::Frame::WORLD_FRAME,
                                (Contacts_[idx].rot * Contacts_[idx].imp) / (world->getTimeStep() + 1e-10),
                                raisim::ArticulatedSystem::Frame::WORLD_FRAME,
                                Contacts_[idx].pos);
      }
    }
  }

  void setTraverseResistiveForce(raisim::World *world, raisim::ArticulatedSystem *robot) {
    if (!inContact_.empty()) {
      for (auto idx: inContact_) {
        if (compliantContacts_[idx].isActive()) {
          if (compliantContacts_[idx].isTraversing()) {
            raisim::Vec<3> F_r;
            raisim::Vec<3> pos_r;
            if (compliantContacts_[idx].getTraverseResistive(&Contacts_[idx], F_r, pos_r)) {
              robot->setExternalForce(footIndices_[idx],
                                      raisim::ArticulatedSystem::Frame::WORLD_FRAME,
                                      F_r,
                                      raisim::ArticulatedSystem::Frame::WORLD_FRAME,
                                      pos_r);
            }
          }
        }
      }
    }
  }

  void maskOnFootCollision(int where_to_mask) {
    for (int i = 0; i < footColObjNames_.size(); ++i) {
      robot_->getCollisionBody(shankColObjNames_[i]).setCollisionMask(raisim::COLLISION(where_to_mask));
      robot_->getCollisionBody(footColObjNames_[i]).setCollisionMask(raisim::COLLISION(where_to_mask));
    }
  }

  void maskOffFootCollision() {
    for (int i = 0; i < footColObjNames_.size(); ++i) {
      robot_->getCollisionBody(shankColObjNames_[i]).setCollisionMask(raisim::COLLISION(i + 2) | raisim::COLLISION(63));
      robot_->getCollisionBody(footColObjNames_[i]).setCollisionMask(raisim::COLLISION(i + 2) | raisim::COLLISION(63));
    }
  }

  void integrate2(raisim::World *world, raisim::ArticulatedSystem *robot) {

    raisim::VecDyn ga(robot->getDOF());
    ga = MinvTauMinusB_;

    for (auto idx: inContact_) {
      raisim::VecDyn v(robot->getDOF());
      sparseJacoTransposeVecmul(Contacts_[idx].sparse_jaco, (Contacts_[idx].rot * Contacts_[idx].imp) / (world->getTimeStep() + 1e-10), v);
      matVecMulThenAdd(M_inv_, v, ga);
    }

    raisim::vecScalarMulThenAdd(world->getTimeStep(), ga, gv_);

    raisim::Vec<4> dq;
    raisim::Vec<4> q_next;
    raisim::Vec<4> q = {gc_[3], gc_[4], gc_[5], gc_[6]};
    raisim::Vec<3> w = {gv_[3], gv_[4], gv_[5]};
    raisim::eulerVecToQuat(w * world->getTimeStep(), dq);
    raisim::quatMul(q, dq, q_next);
    q_next /= q_next.norm();
    for (int i = 0; i < gc_.size(); ++i) {
      if (i == 0 || i == 1 || i == 2) {
        gc_[i] += world->getTimeStep() * gv_[i];
      } else if (i == 3 || i == 4 || i == 5 || i == 6) {
        gc_[i] = q_next[i - 3];
      } else {
        gc_[i] += world->getTimeStep() * gv_[i - 1];
      }
    }

    robot->setGeneralizedCoordinate(gc_);
    robot->setGeneralizedVelocity(gv_);
  }

  void updateContactEligibility() {
    // TODO: TO BE TUNED CORRESPONDING TO THE SIMULATION DT
    for (int i = 0; i < contactEligibilityTrace_.size(); ++i) {
      if (inContact_.count(i)) {
        if (Contacts_[i].imp.e().norm() > 1e-8) {
          contactEligibilityTrace_[i] = 1.0;
        } else {
          contactEligibilityTrace_[i] *= 0.99;
        }
      } else {
        contactEligibilityTrace_[i] *= 0.99;
      }

      if (contactStatus_[i] && contactEligibilityTrace_[i] < 0.8) {
        contactStatus_[i] = false;
      } else if (!contactStatus_[i] && contactEligibilityTrace_[i] >= 0.8) {
        contactStatus_[i] = true;
      }
    }
  }

  void getContactStatus(std::vector<bool> &contactStatus) {
    std::copy(contactStatus_.begin(), contactStatus_.end(), contactStatus.begin());
  }

  double getGroundHeight() {
    if (compliantContacts_.empty()) {
      return 0.;
    }
    double height = 0.;
    for (auto &compliantContact: compliantContacts_) {
      if (compliantContact.isActive()) {
        height = -compliantContact.getMaxPenetrationDepth() < height ?
                 -compliantContact.getMaxPenetrationDepth() : height;
      }
    }
    return height;
  }

  double getMaxPenetrationDepth(int index) {
    return compliantContacts_[index].getMaxPenetrationDepth();
  }

  bool getJacoTransposeGRFPerLeg(size_t idx, raisim::World *world, Eigen::VectorXd &JTF) {
    if (!inContact_.count(idx) || Contacts_[idx].imp.e().norm() < 1e-8) {
      return false;
    } else {
      JTF.segment(3 * idx, 3) = sparseJacoTransposeVecmulPerLeg(Contacts_[idx].sparse_jaco,
                                                                (Contacts_[idx].rot * Contacts_[idx].imp) / (world->getTimeStep() + 1e-10));
      return true;
    }
  }

 private:
  raisim::ArticulatedSystem *robot_;
  std::vector<raisim::Box*> boxes_;

  raisim::VecDyn gc_;
  raisim::VecDyn gv_;

  raisim::MatDyn M_inv_;
  raisim::VecDyn gf_;
  raisim::VecDyn b_;
  raisim::VecDyn tauMinusB_;
  raisim::VecDyn MinvTauMinusB_;

  Eigen::VectorXd PGain_;
  Eigen::VectorXd DGain_;

  std::vector<size_t> footIndices_;
  std::vector<raisim::Vec<3>> footPosition_;
  std::vector<raisim::Vec<3>> footCollisionOffset_;

  std::vector<std::string> shankColObjNames_;
  std::vector<std::string> footColObjNames_;

  std::vector<CompliantContact> compliantContacts_;
  std::vector<ContactContainer> Contacts_;
  std::set<size_t> inContact_;
  std::bitset<4> isRigidContact_;

  int clk_;

  raisim::Vec<4> contactEligibilityTrace_;
  std::vector<bool> contactStatus_;

  GM gm_;

  static double prox_z (const double Imp_z) {
    if (Imp_z < 0) {
      return 0;
    } else {
      return Imp_z;
    }
  }

  static raisim::Vec<2> prox_t (const Eigen::Vector2d & in, const double &lam_z, const double &mu) {
    if (in.norm() <= mu * lam_z) {
      return in;
    } else {
      return ((mu * lam_z) / in.norm()) * in;
    }
  }

  static void sparseJacoTransposeVecmul(const raisim::SparseJacobian &sparse_jaco, const raisim::Vec<3> &vec, raisim::VecDyn &result) {
    result.setZero();
    for (size_t i = 0; i < sparse_jaco.size; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        result[sparse_jaco.idx[i]] += sparse_jaco(j, i) * vec[j];
      }
    }
  }

  static Eigen::Vector3d sparseJacoTransposeVecmulPerLeg(const raisim::SparseJacobian &sparse_jaco, const raisim::Vec<3> &vec) {
    Eigen::Vector3d result;
    result.setZero();
    for (size_t i = sparse_jaco.size - 3; i < sparse_jaco.size; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        result[i - (sparse_jaco.size - 3)] += sparse_jaco(j, i) * vec[j];
      }
    }
    return result;
  }

  static void matSparseJacoTransposeMul(const raisim::MatDyn &M, const raisim::SparseJacobian &jaco, raisim::MatDyn &result) {
    result.setZero();
    for (size_t i = 0; i < M.rows(); ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < jaco.size; ++k) {
          result(i, j) += M(i, jaco.idx[k]) * jaco(j, k);
        }
      }
    }
  }

  static void sparseJacoMatMul(const raisim::SparseJacobian &jaco, const raisim::MatDyn &M, raisim::Mat<3, 3> &result) {
    result.setZero();
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < jaco.size; ++k) {
          result(i, j) += jaco(i, k) * M(jaco.idx[k], j);
        }
      }
    }
  }

  static void matVecMulThenAdd(const raisim::MatDyn &M, const raisim::VecDyn &v, raisim::VecDyn &result) {
    for (size_t i = 0; i < M.rows(); ++i) {
      for (size_t j = 0; j < M.cols(); ++j) {
        result[i] += M(i, j) * v[j];
      }
    }
  }

  static void matTransposeSparseJaco(const raisim::Mat<3, 3> &m, const raisim::SparseJacobian &jaco, raisim::SparseJacobian &result) {
    if (jaco.size != result.size) result.resize(jaco.size);
    std::copy(jaco.idx.begin(), jaco.idx.end(), result.idx.begin());
    result.v.setZero();
    for (size_t j = 0; j < jaco.size; ++j) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t k = 0; k < 3; ++k) {
          result(i, j) += m(k, i) * jaco(k, j);
        }
      }
    }
  }

  static void matvecmul(const raisim::MatDyn &mat1, const raisim::Vec<3> &vec1, raisim::VecDyn &vec) {
    for (size_t j = 0; j < mat1.n; j++) {
      vec[j] = 0;
      for (size_t i = 0; i < mat1.m; i++) // col
        vec[j] += mat1[i * mat1.n + j] * vec1[i];
    }
  }

  static void sparseJacoVecMulThenAdd(const raisim::SparseJacobian &jaco, const raisim::VecDyn &vec, raisim::Vec<3> &result) {
    for(size_t i=0; i< jaco.size; i++) {
      for(size_t j=0; j<3; j++) {
        result[j] += jaco[i*3+j] * vec[jaco.idx[i]];
      }
    }
  }

};

} // namespace raisim

#endif  //RSG_RAIBOT__WORLD_BRIDGE_HPP_