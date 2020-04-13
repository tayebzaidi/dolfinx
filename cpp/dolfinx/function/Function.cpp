// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
#include <cfloat>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::function;

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      la::petsc_error(ierr, __FILE__, NAME);                                   \
  } while (0)

namespace
{
//-----------------------------------------------------------------------------
// Create a vector with layout from dofmap, and zero.
la::PETScVector create_vector(const function::FunctionSpace& V,
                              std::vector<PetscScalar>& vec)
{
  common::Timer timer("Init dof vector");

  // Get dof map
  assert(V.dofmap());
  const fem::DofMap& dofmap = *(V.dofmap());

  // Check that function space is not a subspace (view)
  assert(dofmap.element_dof_layout);
  if (dofmap.element_dof_layout->is_view())
  {
    throw std::runtime_error(
        "Cannot initialize vector of degrees of freedom for "
        "function. Cannot be created from subspace. Consider "
        "collapsing the function space");
  }

  assert(dofmap.index_map);

  auto ghost_indices = dofmap.index_map->ghosts();
  int block_size = dofmap.index_map->block_size();
  int local_size = dofmap.index_map->size_local();
  MPI_Comm comm = dofmap.index_map->mpi_comm();
  std::array<std::int64_t, 2> range = dofmap.index_map->local_range();

  Vec v;
  std::vector<PetscInt> _ghost_indices(ghost_indices.rows());
  for (std::size_t i = 0; i < _ghost_indices.size(); ++i)
    _ghost_indices[i] = ghost_indices(i);
  PetscErrorCode ierr = VecCreateGhostBlockWithArray(
      comm, block_size, block_size * local_size, PETSC_DECIDE,
      _ghost_indices.size(), _ghost_indices.data(), vec.data(), &v);
  CHECK_ERROR("VecCreateGhostBlockWithArray");
  assert(v);

  // Set from PETSc options. This will set the vector type.
  // ierr = VecSetFromOptions(_x);
  // CHECK_ERROR("VecSetFromOptions");

  // NOTE: shouldn't need to do this, but there appears to be an issue
  // with PETSc
  // (https://lists.mcs.anl.gov/pipermail/petsc-dev/2018-May/022963.html)
  // Set local-to-global map
  std::vector<PetscInt> l2g(local_size + ghost_indices.size());
  std::iota(l2g.begin(), l2g.begin() + local_size, range[0]);
  std::copy(ghost_indices.data(), ghost_indices.data() + ghost_indices.size(),
            l2g.begin() + local_size);
  ISLocalToGlobalMapping petsc_local_to_global;
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, block_size, l2g.size(),
                                      l2g.data(), PETSC_COPY_VALUES,
                                      &petsc_local_to_global);
  CHECK_ERROR("ISLocalToGlobalMappingCreate");
  ierr = VecSetLocalToGlobalMapping(v, petsc_local_to_global);
  CHECK_ERROR("VecSetLocalToGlobalMapping");
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global);
  CHECK_ERROR("ISLocalToGlobalMappingDestroy");

  return la::PETScVector(v, true);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V)
    : _id(common::UniqueIdGenerator::id()), _function_space(V),
      _vec(V->dofmap()->index_map->block_size()
               * (V->dofmap()->index_map->size_local()
                  + V->dofmap()->index_map->num_ghosts()),
           0.0),
      _vector(create_vector(*V, _vec))
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    throw std::runtime_error("Cannot create Function from subspace. Consider "
                             "collapsing the function space");
  }
}
//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V,
                   std::vector<PetscScalar> x)
    : _id(common::UniqueIdGenerator::id()), _function_space(V), _vec(x),
      _vector(create_vector(*V, _vec))
{

  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  assert(V->dofmap());
  assert(V->dofmap()->index_map->size_global()
             * V->dofmap()->index_map->block_size()
         <= _vector.size());
}
//-----------------------------------------------------------------------------
Function Function::sub(int i) const
{
  // Extract function subspace
  auto sub_space = _function_space->sub({i});

  // Return sub-function
  assert(sub_space);
  return Function(sub_space, _vec);
}
//-----------------------------------------------------------------------------
Function Function::collapse() const
{
  // Create new collapsed FunctionSpace
  const auto [function_space_new, collapsed_map] = _function_space->collapse();

  // Create new vector
  assert(function_space_new);
  auto im_new = function_space_new->dofmap()->index_map;
  std::vector<PetscScalar> vector_new(
      im_new->block_size() * (im_new->size_local() + im_new->num_ghosts()));

  // Copy values into new vector
  for (std::size_t i = 0; i < collapsed_map.size(); ++i)
  {
    assert(i < vector_new.size());
    assert(collapsed_map[i] < (int)_vec.size());
    vector_new[i] = _vec[collapsed_map[i]];
  }

  return Function(function_space_new, vector_new);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace> Function::function_space() const
{
  return _function_space;
}
//-----------------------------------------------------------------------------
// la::PETScVector& Function::vector()
// {
//   // Check that this is not a sub function.
//   assert(_function_space->dofmap());
//   assert(_function_space->dofmap()->index_map);
//   if (_vector.size()
//       != _function_space->dofmap()->index_map->size_global()
//              * _function_space->dofmap()->index_map->block_size())
//   {
//     throw std::runtime_error(
//         "Cannot access a non-const vector from a subfunction");
//   }

//   return _vector;
// }
//-----------------------------------------------------------------------------
const la::PETScVector& Function::vector() const { return _vector; }
//-----------------------------------------------------------------------------
void Function::eval(
    const Eigen::Ref<
        const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& cells,
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        u) const
{
  // TODO: This could be easily made more efficient by exploiting points
  // being ordered by the cell to which they belong.

  if (x.rows() != cells.rows())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }
  if (x.rows() != u.rows())
  {
    throw std::runtime_error("Length of array for Function values must be the "
                             "same as the number of points.");
  }

  // Get mesh
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Get coordinate map
  const fem::CoordinateElement& cmap = mesh.geometry().cmap();

  // Get element
  assert(_function_space->element());
  const fem::FiniteElement& element = *_function_space->element();
  const int reference_value_size = element.reference_value_size();
  const int value_size = element.value_size();
  const int space_dimension = element.space_dimension();

  // Prepare geometry data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> J(1, gdim, tdim);
  Eigen::Array<double, Eigen::Dynamic, 1> detJ(1);
  Eigen::Tensor<double, 3, Eigen::RowMajor> K(1, tdim, gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(1,
                                                                          tdim);

  // Prepare basis function data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
      1, space_dimension, reference_value_size);
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(1, space_dimension,
                                                         value_size);

  // Create work vector for expansion coefficients
  Eigen::Matrix<PetscScalar, 1, Eigen::Dynamic> coefficients(
      element.space_dimension());

  // Get dofmap
  assert(_function_space->dofmap());
  const fem::DofMap& dofmap = *_function_space->dofmap();

  mesh.topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  // Loop over points
  u.setZero();
  Eigen::Map<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> _v(
      _vec.data(), _vec.size());

  for (Eigen::Index p = 0; p < cells.rows(); ++p)
  {
    const int cell_index = cells(p);

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell_index);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Compute reference coordinates X, and J, detJ and K
    cmap.compute_reference_geometry(X, J, detJ, K, x.row(p).head(gdim),
                                    coordinate_dofs);

    // Compute basis on reference element
    element.evaluate_reference_basis(basis_reference_values, X);

    // Push basis forward to physical element
    element.transform_reference_basis(basis_values, basis_reference_values, X,
                                      J, detJ, K, cell_info[cell_index]);

    // Get degrees of freedom for current cell
    auto dofs = dofmap.cell_dofs(cell_index);
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
      coefficients[i] = _v[dofs[i]];

    // Compute expansion
    for (int i = 0; i < space_dimension; ++i)
    {
      for (int j = 0; j < value_size; ++j)
      {
        // TODO: Find an Eigen shortcut for this operation
        u.row(p)[j] += coefficients[i] * basis_values(0, i, j);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(const Function& v)
{
  assert(_function_space);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x(_vec.data(),
                                                             _vec.size());
  _function_space->interpolate(x, v);
}
//-----------------------------------------------------------------------------
void Function::interpolate(
    const std::function<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& f)
{
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x(_vec.data(),
                                                             _vec.size());
  _function_space->interpolate(x, f);
}
//-----------------------------------------------------------------------------
void Function::interpolate_c(const FunctionSpace::interpolation_function& f)
{

  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x(_vec.data(),
                                                             _vec.size());
  _function_space->interpolate_c(x, f);
}
//-----------------------------------------------------------------------------
int Function::value_rank() const
{
  assert(_function_space);
  assert(_function_space->element());
  return _function_space->element()->value_rank();
}
//-----------------------------------------------------------------------------
int Function::value_size() const
{
  int size = 1;
  for (int i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
int Function::value_dimension(int i) const
{
  assert(_function_space);
  assert(_function_space->element());
  return _function_space->element()->value_dimension(i);
}
//-----------------------------------------------------------------------------
std::vector<int> Function::value_shape() const
{
  std::vector<int> _shape(this->value_rank(), 1);
  for (std::size_t i = 0; i < _shape.size(); ++i)
    _shape[i] = this->value_dimension(i);
  return _shape;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Function::compute_point_values() const
{
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  const int tdim = mesh.topology().dim();

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding point values
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      point_values(mesh.geometry().x().rows(), value_size_loc);

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Interpolate point values on each cell (using last computed value if
  // not continuous, e.g. discontinuous Galerkin methods)
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x(num_dofs_g, 3);
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(num_dofs_g, value_size_loc);
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get coordinates for all points in cell
    auto dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      x.row(i) = x_g.row(dofs[i]);

    values.resize(x.rows(), value_size_loc);

    // Call evaluate function
    Eigen::Array<int, Eigen::Dynamic, 1> cells(x.rows());
    cells = c;
    eval(x, cells, values);

    // Copy values to array of point values
    for (int i = 0; i < x.rows(); ++i)
      point_values.row(dofs[i]) = values.row(i);
  }

  return point_values;
}
//-----------------------------------------------------------------------------
std::size_t Function::id() const { return _id; }
//-----------------------------------------------------------------------------
