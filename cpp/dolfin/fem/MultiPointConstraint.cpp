// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfin/fem/DofMap.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/MeshIterator.h>

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

/// Partition cells into cells containing slave dofs and cells with no cell dofs
std::pair<std::vector<int>, std::vector<int>>
MultiPointConstraint::cell_classification()
{
  const mesh::Mesh& mesh = *(_function_space->mesh());
  const fem::DofMap& dofmap = *(_function_space->dofmap());
  std::vector<int> slaves = {};
  std::vector<int> normals = {};

  // Categorise cells as normal cells or slave cells, which can later be used in
  // the custom assembly
  for (auto& cell : mesh::MeshRange(mesh, mesh.topology().dim()))
  {
    const int cell_index = cell.index();
    auto dofs = dofmap.cell_dofs(cell_index);
    bool slave_cell = false;
    for (auto slave = _slave_to_master.begin(); slave != _slave_to_master.end();
         slave++)
    {
      for (Eigen::Index i = 0; i < dofs.size(); ++i)
      {
        if (unsigned(dofs[i]) == slave->first)
        {
          slave_cell = true;
        }
      }
    }
    if (slave_cell)
    {
      slaves.push_back(cell_index);
    }
    else
    {
      normals.push_back(cell_index);
    }
  }
  return std::pair(slaves, normals);
}
