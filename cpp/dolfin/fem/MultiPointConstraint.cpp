// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include <dolfin/function/FunctionSpace.h>

using namespace dolfin;
using namespace dolfin::fem;

MultiPointConstraint::MultiPointConstraint(
    std::shared_ptr<const function::FunctionSpace> V,
    std::unordered_map<std::size_t, std::size_t> slave_to_master)
    : _function_space(V), _slave_to_master(slave_to_master)
{
}

/// Slave to master map
std::unordered_map<std::size_t, std::size_t>
MultiPointConstraint::slave_to_master() const
{
  return _slave_to_master;
}
