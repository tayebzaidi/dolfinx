// Copyright (C) 2017 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
#include "Mesh.h"
#include "MeshFunction.h"
#include "MeshIterator.h"
#include "MeshPartitioning.h"

#include "MeshView.h"

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
Mesh MeshView::create(const MeshFunction<std::size_t>& marker, std::size_t tag)
{
  // Get original Mesh and tdim of marker
  std::shared_ptr<const Mesh> mesh = marker.mesh();
  unsigned int tdim = marker.dim();

  // Get indices of marked entities - each of these will represent a Cell in
  // "new_mesh"
  std::vector<std::size_t> indices;
  for (std::int64_t idx = 0; idx < mesh->num_entities(tdim); ++idx)
  {
    if (marker[idx] == tag)
      indices.push_back(idx);
  }

  // Create a new Mesh of dimension tdim
  std::unique_ptr<CellType> entity_type(
      CellType::create(mesh->type().entity_type(tdim)));
  std::size_t num_cell_vertices = entity_type->num_vertices();

  // Reverse mapping from full Mesh for vertices
  std::map<std::size_t, std::size_t> vertex_rev_map;
  // Forward map to full Mesh for vertices
  std::vector<std::size_t> vertex_fwd_map;

  // Establish a new local numbering for all vertices in "new_mesh"
  // and create the mapping back to the local numbering in "mesh"

  std::size_t vertex_num = 0;
  const auto& conn_tdim = mesh->topology().connectivity(tdim, 0);

  // Add cells to new_mesh
  EigenRowArrayXXi64 new_cells(indices.size(), num_cell_vertices);
  for (unsigned int i = 0; i != indices.size(); ++i)
  {
    const std::size_t idx = indices[i];
    for (unsigned int j = 0; j != num_cell_vertices; ++j)
    {
      const std::size_t main_idx = conn_tdim(idx)[j];
      auto mapit = vertex_rev_map.insert({main_idx, vertex_num});
      if (mapit.second)
      {
        vertex_fwd_map.push_back(main_idx);
        ++vertex_num;
      }
      new_cells(i, j) = mapit.first->second;
    }
  }

  // Generate some global indices for new cells
  const std::size_t cell_offset
      = MPI::global_offset(mesh->mpi_comm(), indices.size(), true);
  std::vector<std::int64_t> new_cell_indices(indices.size());
  std::iota(new_cell_indices.begin(), new_cell_indices.end(), cell_offset);

  // Initialise vertex global indices in new_mesh
  const std::size_t mpi_size = MPI::size(mesh->mpi_comm());
  const std::size_t mpi_rank = MPI::rank(mesh->mpi_comm());

  // Create sharing map from main mesh data
  std::map<std::int32_t, std::set<std::uint32_t>> new_shared;
  const auto& mesh_shared = mesh->topology().shared_entities(0);
  const auto& mesh_global = mesh->topology().global_indices(0);

  // Global numbering in new_mesh. Shared vertices are numbered by
  // the lowest rank sharing process
  std::size_t local_count = 0;
  std::vector<std::size_t> vertex_global_index(vertex_num);
  std::vector<std::vector<std::size_t>> send_vertex_numbering(mpi_size);
  std::vector<std::vector<std::size_t>> recv_vertex_numbering(mpi_size);

  // Map from global vertices on main mesh to local on new_mesh
  std::map<std::size_t, std::size_t> main_global_to_new;

  for (unsigned int i = 0; i != vertex_num; ++i)
  {
    const std::size_t main_idx = vertex_fwd_map[i];
    const auto shared_it = mesh_shared.find(main_idx);
    if (shared_it == mesh_shared.end())
    {
      // Vertex not shared in main mesh, so number locally
      vertex_global_index[i] = local_count;
      ++local_count;
    }
    else
    {
      new_shared.insert({i, shared_it->second});
      // Send to remote 'owner' for numbering.
      std::size_t dest = *(shared_it->second.begin());
      if (dest > mpi_rank)
      {
        // Shared, but local - number locally
        vertex_global_index[i] = local_count;
        ++local_count;
      }
      else
        send_vertex_numbering[dest].push_back(mesh_global[main_idx]);

      main_global_to_new.insert({mesh_global[main_idx], i});
    }
  }

  // Send global indices of all unnumbered vertices to owner
  MPI::all_to_all(mesh->mpi_comm(), send_vertex_numbering,
                  recv_vertex_numbering);

  // Create global->local map for *all* shared vertices of main mesh
  std::map<std::size_t, std::size_t> main_global_to_local;
  for (auto it : mesh_shared)
    main_global_to_local.insert({mesh_global[it.first], it.first});

  // Search for received vertices which are not already there. This may be
  // because they are not part of a new_mesh cell for this process.
  for (auto& p : recv_vertex_numbering)
  {
    for (auto& q : p)
    {
      if (main_global_to_new.find(q) == main_global_to_new.end())
      {
        // Not found - shared, but not part of local MeshView, yet.
        // This Vertex may have no locally associated Cell in new_mesh
        auto mgl_find = main_global_to_local.find(q);
        assert(mgl_find != main_global_to_local.end());
        const std::size_t main_idx = mgl_find->second;
        vertex_fwd_map.push_back(main_idx);
        vertex_global_index.push_back(local_count);
        main_global_to_new.insert({q, vertex_num});
        const auto shared_it = mesh_shared.find(main_idx);
        assert(shared_it != mesh_shared.end());
        new_shared.insert({vertex_num, shared_it->second});
        ++local_count;
        ++vertex_num;
      }
    }
  }

  // All shared vertices are now numbered

  // Add geometry and create new_mesh
  EigenRowArrayXXd new_points(vertex_fwd_map.size(), mesh->geometry().dim());
  for (unsigned int i = 0; i != vertex_fwd_map.size(); ++i)
    new_points.row(i) = mesh->geometry().x(vertex_fwd_map[i]);

  Mesh new_mesh(mesh->mpi_comm(), entity_type->cell_type(), new_points,
                new_cells, new_cell_indices, mesh::GhostMode::none, 0);

  auto& new_shared_0 = new_mesh.topology().shared_entities(0);
  new_shared_0.insert(new_shared.begin(), new_shared.end());

  // Correct for global vertex index offset
  const std::size_t vertex_offset
      = MPI::global_offset(mesh->mpi_comm(), local_count, true);
  for (auto& v : vertex_global_index)
    v += vertex_offset;

  // Convert incoming main global index to new_mesh global index
  for (auto& p : recv_vertex_numbering)
    for (auto& q : p)
    {
      const auto map_it = main_global_to_new.find(q);
      assert(map_it != main_global_to_new.end());
      q = vertex_global_index[map_it->second];
    }

  // Send reply back to originator
  std::vector<std::vector<std::size_t>> reply_vertex_numbering(mpi_size);
  MPI::all_to_all(mesh->mpi_comm(), recv_vertex_numbering,
                  reply_vertex_numbering);

  // Convert main global back to new_mesh local and save new_mesh global
  // index
  for (unsigned int i = 0; i != mpi_size; ++i)
  {
    std::vector<std::size_t>& send_v = send_vertex_numbering[i];
    std::vector<std::size_t>& reply_v = reply_vertex_numbering[i];
    assert(send_v.size() == reply_v.size());

    for (unsigned int j = 0; j != send_v.size(); ++j)
    {
      auto map_it = main_global_to_new.find(send_v[j]);
      assert(map_it != main_global_to_new.end());
      vertex_global_index[map_it->second] = reply_v[j];
    }
  }

  MeshTopology& new_topo = new_mesh.topology();

  // Set global vertex indices
  new_topo.init_global_indices(0, vertex_num);
  for (std::size_t i = 0; i != vertex_num; ++i)
    new_topo.set_global_index(0, i, vertex_global_index[i]);

  // Store relationship between meshes
  //  new_topo._mapping = std::make_shared<MeshView>(mesh, vertex_fwd_map,
  //  indices);

  return new_mesh;
}
