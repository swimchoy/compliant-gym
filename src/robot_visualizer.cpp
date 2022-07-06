// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "worldBridge.hpp"
#include "controller/raibot_compliant_controller/raibot_compliant_controller.hpp"

int main (int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  /// create world and robot
  double control_dt = 0.01;
  double simulation_dt = 0.00025;

  raisim::World world;
  world.setTimeStep(simulation_dt);

  auto robot = world.addArticulatedSystem(binaryPath.getDirectory() + "/rsc/raibot/urdf/raibot_simplified.urdf");
  robot->setName("robot");
  robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

  /// set ground contacting with body and thighs
  std::vector<std::string> bodyColObjNames = {"base/0", "LF_THIGH/0", "RF_THIGH/0", "LH_THIGH/0", "RH_THIGH/0"};
  for (auto &name: bodyColObjNames) { robot->getCollisionBody(name).setCollisionGroup(raisim::COLLISION(10)); }
  world.setDefaultMaterial(0.15, 0.0, 0.01);
  world.addGround(0.0, "default", raisim::COLLISION(10));
//  world.addGround(0.0, "default", raisim::COLLISION(2) | raisim::COLLISION(3) | raisim::COLLISION(4) | raisim::COLLISION(5));

  raisim::worldBridge bridge;
  controller::raibotCompliantController controller;

  bridge.create(&world);
  controller.create(&world);
  controller.setTimeConfig(control_dt, simulation_dt);
  controller.reset(&world);
  bridge.reset();

  raisim::GM gm;
  gm.k_div_A = 1.0e6;
  bridge.setTerrainParameters(gm);

  Eigen::Vector3f command = {1.0, 0.0, 0.0};
  controller.setCommand(command);

  raisim::RaisimServer server(&world);
  server.launchServer(8080);
  server.focusOn(robot);

  /// simulation
  int maxStep = 100000000;
  for (int i = 0; i < maxStep; ++i) {
    controller.advance(&world);
    bridge.integrate(&world);
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  server.killServer();
}