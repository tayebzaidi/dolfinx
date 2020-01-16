// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/SparsityPattern.h>
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

// Append to existing sparsity pattern
std::shared_ptr<dolfin::la::SparsityPattern>
MultiPointConstraint::generate_sparsity_pattern(
    const Form& a, std::shared_ptr<dolfin::la::SparsityPattern> pattern)
{
  std::array<const DofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};
  dofmaps[0];
  dofmaps[1];
  const std::unordered_map<std::size_t, std::size_t> pairs = _slave_to_master;

  auto mesh = *_function_space->mesh();
  const int D = mesh.topology().dim();
  for (auto& cell : mesh::MeshRange(mesh, D))
  {

    // Master slave sparsity pattern
    std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> master_slave_dofs;
    // Dofs previously owned by slave dof
    std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> new_master_dofs;

    for (std::size_t i = 0; i < 2; i++)
    {
      auto cell_dof_list = dofmaps[i]->cell_dofs(cell.index());
      new_master_dofs[i].resize(cell_dof_list.size());
      std::copy(cell_dof_list.data(),
                cell_dof_list.data() + cell_dof_list.size(),
                new_master_dofs[i].data());
      for (auto it = pairs.begin(); it != pairs.end(); ++it)
      {
        for (std::size_t j = 0; j < unsigned(cell_dof_list.size()); ++j)
        {

          if (it->first == unsigned(cell_dof_list[j]))
          {
            new_master_dofs[i](j) = it->second;
            master_slave_dofs[i].conservativeResize(master_slave_dofs[i].size()
                                                    + 2);
            master_slave_dofs[i].row(master_slave_dofs[i].rows() - 1)
                = it->first;
            master_slave_dofs[i].row(master_slave_dofs[i].rows() - 2)
                = it->second;
          }
        }
      }
    }
    pattern->insert_local(new_master_dofs[0], new_master_dofs[1]);
    pattern->insert_local(master_slave_dofs[0], master_slave_dofs[1]);
  }
  pattern->assemble();
  return pattern;
}
