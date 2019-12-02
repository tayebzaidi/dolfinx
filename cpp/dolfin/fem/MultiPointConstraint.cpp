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
    std::unordered_map<std::size_t, std::size_t> master_slave_map)
    : _function_space(V), _master_slave_map(master_slave_map)
{
}
