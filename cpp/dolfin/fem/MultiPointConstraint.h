// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/fem/Form.h>

namespace dolfin
{
namespace fem
{

  /// This class provides the interface for setting multi-point constraints.
  ///
  ///   u_i = u_j,
  ///
  /// where u_i and u_j denotes the i-th and j-th global degree of freedom in the corresponding
  /// function space.
  /// A MultiPointBC is specified by the function space (trial space), and a vector of pairs, connecting
  /// master and slave nodes with a linear dependency.

  class MultiPointConstraint
  {

  public:
  /// Create multipointconstraint
  ///
  /// @param[in] V The functionspace on which the multipoint constraint
  /// condition is applied
  /// @param[in] slave_to_master Mapping specifying the relationship between a slave and a master node.
	MultiPointConstraint(std::shared_ptr<const function::FunctionSpace> V,  std::unordered_map<std::size_t, std::size_t> slave_to_master);

	/// Slave to master map
	std::unordered_map<std::size_t, std::size_t> slave_to_master() const;

	//Return two arrays, where the first contain cell indices of all cells containing cell dofs, and the second cell containing the others
	std::pair<std::vector<int>, std::vector<int>>
	  cell_classification();


	// Add sparsity pattern for multi-point constraints to existing sparsity pattern
	std::shared_ptr<la::SparsityPattern> generate_sparsity_pattern(const Form&,std::shared_ptr<la::SparsityPattern>);

  private:
	std::shared_ptr<const function::FunctionSpace> _function_space;
	const std::unordered_map<std::size_t, std::size_t> _slave_to_master;
   };

}
}
