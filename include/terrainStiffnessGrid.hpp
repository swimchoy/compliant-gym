// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef _COMPLIANT_GYM__TERRAIN_STIFFNESS_GRID_HPP_
#define _COMPLIANT_GYM__TERRAIN_STIFFNESS_GRID_HPP_

#include <Eigen/Core>
#include "grid/grid.hpp"

namespace env {

class terrainStiffnessGrid {

 public:
  terrainStiffnessGrid();
  void reset();
  bool update();
  Eigen::VectorXd sample();
  size_t getNumTotalGrid() const;

 private:
  size_t iteration_;
  size_t numTotalGrid_;
  std::vector<std::string> propertyNames_;
  std::map<std::string, Grid<double>> grid_;

};

terrainStiffnessGrid::terrainStiffnessGrid()
{
  propertyNames_ = {"sigma_flat", "sigma_cone"};

  double deg_to_rad = 3.14159 / 180.;
  grid_.insert(std::make_pair("sigma_flat", Grid<double>(1.0e6, 1.0e7, 10)));
  grid_.insert(std::make_pair("sigma_cone", Grid<double>(0.15e6, 0.6e6, 10)));

  iteration_ = 0;
  numTotalGrid_ = 1;
  for (auto [_, grid]: grid_) {
    numTotalGrid_ *= grid.numGrid_;
  }
}

Eigen::VectorXd terrainStiffnessGrid::sample()
{
  Eigen::VectorXd p(10);
  p << 2.0, 3.14159 / 3., 2.7, 0.57, 1000, 1.0e6, 17.2, 0.15e6, 160.0, 1.0;

  p[5] = grid_.at("sigma_flat").get();
  p[7] = grid_.at("sigma_cone").get();

  return p;
}

void terrainStiffnessGrid::reset() {
  iteration_ = 0;
  for (auto [_, grid]: grid_) {
    grid.idx_ = 0;
  }
}

bool terrainStiffnessGrid::update()
{
  if (iteration_ + 1 > numTotalGrid_) {
    return false;
  } else {
    ++iteration_;
  }

  for (auto &propertyName: propertyNames_) {
    if (grid_.at(propertyName).update()) break;
  }

  return true;
}

size_t terrainStiffnessGrid::getNumTotalGrid() const
{
  return numTotalGrid_;
}

} // namespace env

#endif //_COMPLIANT_GYM__TERRAIN_STIFFNESS_GRID_HPP_
