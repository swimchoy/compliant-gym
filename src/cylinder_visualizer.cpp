// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

#include "addedMass.hpp"
#include "terrainStiffnessGrid.hpp"

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  /// config world
  double simulation_dt = 0.00025;
  raisim::World world;
  world.setTimeStep(simulation_dt);

  /// config visualizer
  raisim::RaisimServer server(&world);
  server.launchServer(8080);

  /// terrain property grid setup
  env::terrainStiffnessGrid grid;
  grid.reset();

  /// for each parameter set, make cylinder and corresponding contact model.
  std::vector<raisim::Cylinder*> cys;
  std::map<size_t, std::shared_ptr<raisim::GM>> gms;
  std::map<size_t, std::shared_ptr<cylinderContact::addedMass>> ams;
  std::map<size_t, raisim::Visuals*> mes;
  Eigen::Matrix3d rot;
  rot << 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0;
  size_t row = grid.getNumTotalGrid() / 10;
  double scale = 0.1;

  for (size_t i = 0; i < grid.getNumTotalGrid(); ++i)
  {
    cys.push_back(world.addCylinder(FOOT::r, FOOT::h, 5));
    cys.back()->setPosition(scale * static_cast<double>(i / row),
                            scale * static_cast<double>(i % row),
                            0.05);
    cys.back()->setOrientation(rot);
    gms[cys.back()->getIndexInWorld()] = std::make_shared<GM>(grid.sample());
    ams[cys.back()->getIndexInWorld()] = std::make_shared<cylinderContact::addedMass>();
    mes[cys.back()->getIndexInWorld()] = server.addVisualMesh(
        std::to_string(i), std::string(binaryPath.getDirectory()) + "/rsc/raibot/meshes/RH_FOOT.STL",
        {1.1, 1.1, 1.1}, 0, 0, 0, 0.65);
    grid.update();
  }

  /// for sandy terrain visualization
  {
    double vboxOffset = 0.5;
    auto vxr = std::make_pair<double, double>(
        scale * -vboxOffset, scale * (static_cast<double>((grid.getNumTotalGrid() - 1) / row) + vboxOffset));
    auto vyr = std::make_pair<double, double>(
        scale * -vboxOffset, scale * (static_cast<double>((grid.getNumTotalGrid() - 1) % row) + vboxOffset));

    auto vGround = server.addVisualBox("ground", vxr.second - vxr.first,
                                    vyr.second - vyr.first,0.001,
                                    0.7765, 0.5412, 0.0706, 0.75);
    vGround->setPosition(0.5 * (vxr.second + vxr.first), 0.5 * (vyr.second + vyr.first), 0.0);
  }

  /// contact model simulation
  while (world.getWorldTime() < 100.0)
  {
    for (auto &c: cys)
    {
      ams[c->getIndexInWorld()]->advance(
          &world, c, gms[c->getIndexInWorld()].get());
      mes[c->getIndexInWorld()]->setPosition(c->getPosition());
    }

    world.integrate();
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
  }

  server.killServer();
}