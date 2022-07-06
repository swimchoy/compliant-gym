// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef _COMPLIANT_GYM__GRID_HPP_
#define _COMPLIANT_GYM__GRID_HPP_

#include <map>
#include <vector>
#include <string>

using namespace raisim;

namespace env {

template <typename dType>
class Grid {

 public:
  size_t idx_;
  size_t numGrid_;

  Grid(dType minValue, dType maxValue, size_t numGrid);
  bool update();
  dType get();

 private:
  std::vector<dType> values_;

};

template <typename dType>
Grid<dType>::Grid(dType minValue, dType maxValue, size_t numGrid)
    : idx_(0)
{
  numGrid_ = numGrid;
  values_.resize(numGrid);
  dType increment = (maxValue - minValue) / static_cast<dType>(numGrid - 1);
  dType value = minValue;
  for (auto x = values_.begin(); x != values_.end(); ++x) {
    *x = value;
    value += increment;
  }
}

template <typename dType>
bool Grid<dType>::update()
{
  if (++idx_ == numGrid_) {
    idx_ = 0;
    return false;
  }
  return true;
}

template <typename dType>
dType Grid<dType>::get() {
  return values_[idx_];
}

} // namespace env

#endif //_COMPLIANT_GYM__GRID_HPP_
