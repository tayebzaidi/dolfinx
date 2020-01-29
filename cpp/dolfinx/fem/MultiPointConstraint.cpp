// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN-X (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/MeshIterator.h>

using namespace dolfinx;
using namespace dolfinx::fem;

MultiPointConstraint::MultiPointConstraint(
    std::shared_ptr<const function::FunctionSpace> V,
    std::vector<std::int64_t> slaves, std::vector<std::int64_t> masters,
    std::vector<double> coefficients, std::vector<std::int64_t> offsets)
    : _function_space(V), _slaves(slaves), _masters(masters),
      _coefficients(coefficients), _offsets(offsets), _slave_cells(),
      _normal_cells(), _offsets_cell_to_slave(), _cell_to_slave()
{
}

/// Get the master nodes for the i-th slave
std::vector<std::int64_t> MultiPointConstraint::masters(std::int64_t i)
{
  assert(i < unsigned(_masters.size()));
  std::vector<std::int64_t> owners;
  // Check if this is the final entry or beyond
  for (std::int64_t j = _offsets[i]; j < _offsets[i + 1]; j++)
  {
    owners.push_back(_masters[j]);
  }
  return owners;
}

/// Get the master nodes for the i-th slave
std::vector<double> MultiPointConstraint::coefficients(std::int64_t i)
{
  assert(i < unsigned(_coefficients.size()));
  std::vector<double> coeffs;
  for (std::int64_t j = _offsets[i]; j < _offsets[i + 1]; j++)
  {
    coeffs.push_back(_coefficients[j]);
  }
  return coeffs;
}

std::vector<std::int64_t> MultiPointConstraint::slave_cells()
{
  return _slave_cells;
}

/// Partition cells into cells containing slave dofs and cells with no cell
/// dofs
std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
MultiPointConstraint::cell_classification()
{
  const mesh::Mesh& mesh = *(_function_space->mesh());
  const fem::DofMap& dofmap = *(_function_space->dofmap());
  // const std::vector<int64_t>& global_indices
  //     = mesh.topology().global_indices(mesh.topology().dim());

  // Categorise cells as normal cells or slave cells,
  // which can later be used in the custom assembly
  std::int64_t j = 0;
  _offsets_cell_to_slave.push_back(j);
  for (auto& cell : mesh::MeshRange(mesh, mesh.topology().dim()))
  {
    const int cell_index = cell.index();
    auto dofs = dofmap.cell_dofs(cell_index);
    bool slave_cell = false;

    for (auto slave : _slaves)
    {
      for (Eigen::Index i = 0; i < dofs.size(); ++i)
      {
        if (unsigned(dofs[i] + dofmap.index_map->local_range()[0]) == slave)
        {
          _cell_to_slave.push_back(slave);
          j++;
          slave_cell = true;
        }
      }
    }
    if (slave_cell)
    {
      _slave_cells.push_back(cell_index);
      _offsets_cell_to_slave.push_back(j);
    }
    else
    {
      _normal_cells.push_back(cell_index);
    }
  }
  return std::pair(_slave_cells, _normal_cells);
}

std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
MultiPointConstraint::cell_to_slave_mapping()
{
  return std::pair(_cell_to_slave, _offsets_cell_to_slave);
}

// Append to existing sparsity pattern
dolfinx::la::SparsityPattern
MultiPointConstraint::generate_sparsity_pattern(const Form& a)
{

  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }
  // Get dof maps
  std::array<const DofMap*, 2> dofmaps
      = {{a.function_space(0)->dofmap().get(),
          a.function_space(1)->dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const mesh::Mesh& mesh = *(a.mesh());

  // Get common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map, dofmaps[1]->index_map}};

  // Need to create new IndexMaps here with additional ghosts corresponding to
  // masters and slaves on different processors

  // Create and build sparsity pattern
  la::SparsityPattern pattern(mesh.mpi_comm(), index_maps);
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
    SparsityPatternBuilder::cells(pattern, mesh, {{dofmaps[0], dofmaps[1]}});
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::interior_facet) > 0)
    SparsityPatternBuilder::interior_facets(pattern, mesh,
                                            {{dofmaps[0], dofmaps[1]}});
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
    SparsityPatternBuilder::exterior_facets(pattern, mesh,
                                            {{dofmaps[0], dofmaps[1]}});

  pattern.info_statistics();

  // Loop over slave cells
  for (std::int64_t i = 0; i < unsigned(_slave_cells.size()); i++)
  {
    std::vector<std::int64_t> slaves_i(
        _cell_to_slave.begin() + _offsets_cell_to_slave[i],
        _cell_to_slave.begin() + _offsets_cell_to_slave[i + 1]);
    // Loop over slaves in cell
    for (auto slave_dof : slaves_i)
    {
      std::int64_t slave_index = 0; // Index in slave array
      // FIXME: Move this somewhere else as there should exist a map for this
      for (std::uint64_t counter = 0; counter < _slaves.size(); counter++)
      {
        if (_slaves[counter] == slave_dof)
        {
          slave_index = counter;
        }
      }
      std::vector<std::int64_t> masters_i(
          _masters.begin() + _offsets[slave_index],
          _masters.begin() + _offsets[slave_index + 1]);
      // Insert pattern for each master
      for (auto master : masters_i)
      {
        // New sparsity pattern arrays
        std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2>
            new_master_dofs;
        // Sparsity pattern needed for columns
        std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2>
            master_slave_dofs;
        // Loop over test and trial space
        for (std::size_t j = 0; j < 2; j++)
        {

          auto cell_dof_list = dofmaps[j]->cell_dofs(_slave_cells[i]);
          std::uint64_t local_min = dofmaps[j]->index_map->local_range()[0];
          new_master_dofs[j].resize(cell_dof_list.size());
          std::copy(cell_dof_list.data(),
                    cell_dof_list.data() + cell_dof_list.size(),
                    new_master_dofs[j].data());

          // Replace slave dof with master dof (global insert)
          for (std::size_t k = 0; k < unsigned(cell_dof_list.size()); ++k)
          {
            if (_slaves[slave_index] == unsigned(cell_dof_list[k] + local_min))
            {
              new_master_dofs[j](k) = master;
              master_slave_dofs[j].conservativeResize(
                  master_slave_dofs[j].size() + 2);
              master_slave_dofs[j].row(master_slave_dofs[j].rows() - 1)
                  = _slaves[slave_index];
              master_slave_dofs[j].row(master_slave_dofs[j].rows() - 2)
                  = master;
            }
            else
            {
              new_master_dofs[j](k) = new_master_dofs[j](k) + local_min;
            }
          }
        }
        // Since all indices are global we need an identity map;
        const auto glob_map
            = [](const PetscInt j_index,
                 const common::IndexMap& index_map1) -> PetscInt {
          return j_index;
        };
        pattern.insert_entries(new_master_dofs[0], new_master_dofs[1], glob_map,
                               glob_map);
        pattern.insert_entries(master_slave_dofs[0], master_slave_dofs[1],
                               glob_map, glob_map);
      }
    }
  }
  pattern.info_statistics();
  pattern.assemble();
  return pattern;
}

std::vector<std::int64_t> MultiPointConstraint::slaves() { return _slaves; }

std::pair<std::vector<std::int64_t>, std::vector<double>>
MultiPointConstraint::masters_and_coefficients()
{
  return std::pair(_masters, _coefficients);
}

/// Return master offset data
std::vector<std::int64_t> MultiPointConstraint::master_offsets()
{
  return _offsets;
}
