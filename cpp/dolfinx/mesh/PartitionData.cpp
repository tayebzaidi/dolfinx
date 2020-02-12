// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PartitionData.h"
#include <algorithm>
#include <sstream>

extern "C"
{
#include <ptscotch.h>
}

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
PartitionData::PartitionData(
    const std::vector<int>& cell_partition,
    const std::map<std::int64_t, std::vector<int>>& ghost_procs)
    : _offset(1)

{
  for (std::size_t i = 0; i < cell_partition.size(); ++i)
  {
    auto it = ghost_procs.find(i);
    if (it == ghost_procs.end())
      _dest_processes.push_back(cell_partition[i]);
    else
    {
      _dest_processes.insert(_dest_processes.end(), it->second.begin(),
                             it->second.end());
    }
    _offset.push_back(_dest_processes.size());
  }
}
//-----------------------------------------------------------------------------
PartitionData::PartitionData(
    const std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>&
        data)
    : PartitionData(data.first, data.second)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::int32_t PartitionData::num_procs(std::int32_t i) const
{
  return _offset[i + 1] - _offset[i];
}
//-----------------------------------------------------------------------------
const std::int32_t* PartitionData::procs(std::int32_t i) const
{
  return _dest_processes.data() + _offset[i];
}
//-----------------------------------------------------------------------------
std::int32_t PartitionData::size() const { return _offset.size() - 1; }
//-----------------------------------------------------------------------------
std::int32_t PartitionData::num_ghosts() const
{
  return _dest_processes.size() - _offset.size() + 1;
}
//-----------------------------------------------------------------------------
void PartitionData::graph(MPI_Comm mpi_comm)
{
  const int mpi_size = MPI::size(mpi_comm);
  const int mpi_rank = MPI::rank(mpi_comm);

  // Make map of connections between processes {proc1, proc2} -> num_connections
  std::map<std::pair<int, int>, int> neighbour_info;
  for (std::size_t i = 0; i < _offset.size() - 1; ++i)
  {
    if (_offset[i + 1] - _offset[i] > 1)
    {
      for (int j = _offset[i]; j < _offset[i + 1]; ++j)
        for (int k = j + 1; k < _offset[i + 1]; ++k)
        {
          std::pair<int, int> idx(_dest_processes[j], _dest_processes[k]);
          neighbour_info[idx]++;
          idx = {idx.second, idx.first};
          neighbour_info[idx]++;
        }
    }
  }

  // Accumulate all connectivity on process 0
  std::vector<int> send_data;
  std::vector<int> recv_data;
  for (auto& info : neighbour_info)
  {
    send_data.push_back(info.first.first);
    send_data.push_back(info.first.second);
    send_data.push_back(info.second);
  }
  MPI::gather(mpi_comm, send_data, recv_data);

  MPI_Comm shmComm;
  MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shmComm);

  std::cout << "shmComm = " << MPI::rank(shmComm) << "/" << MPI::size(shmComm)
            << "\n";

  const int rank0 = (int)(MPI::rank(shmComm) == 0);
  const int nnodes = MPI::sum(mpi_comm, rank0);

  std::vector<int> renumbering(mpi_size);

  if (mpi_rank == 0)
  {
    std::vector<std::map<int, int>> edgeconn(mpi_size);
    for (std::size_t i = 0; i < recv_data.size(); i += 3)
    {
      std::map<int, int>& ec = edgeconn[recv_data[i]];
      auto it = ec.find(recv_data[i + 1]);
      if (it == ec.end())
        ec.insert({recv_data[i + 1], recv_data[i + 2]});
      else
        it->second += recv_data[i + 2];
    }

    std::vector<SCOTCH_Num> vertloctab = {0};
    std::vector<SCOTCH_Num> edgeloctab;
    std::vector<SCOTCH_Num> edloloctab;
    for (auto& node : edgeconn)
    {
      for (const auto& remote : node)
      {
        edgeloctab.push_back(remote.first);
        edloloctab.push_back(remote.second);
      }
      vertloctab.push_back(edgeloctab.size());
    }
    assert((int)vertloctab.size() == mpi_size + 1);
    const SCOTCH_Num vertlocnbr = mpi_size;

    std::cout << "vertloctab =";
    for (auto q : vertloctab)
      std::cout << q << ",";
    std::cout << "\n";

    std::cout << "edgeloctab =";
    for (auto q : edgeloctab)
      std::cout << q << ",";
    std::cout << "\n";

    std::cout << "edloloctab =";
    for (auto q : edloloctab)
      std::cout << q << ",";
    std::cout << "\n";

    SCOTCH_Dgraph dgrafdat;
    if (SCOTCH_dgraphInit(&dgrafdat, MPI_COMM_SELF) != 0)
      throw std::runtime_error("Error initializing SCOTCH graph");

    if (SCOTCH_dgraphBuild(
            &dgrafdat, 0, vertlocnbr, vertlocnbr,
            const_cast<SCOTCH_Num*>(vertloctab.data()), nullptr, nullptr,
            nullptr, edgeloctab.size(), edgeloctab.size(),
            const_cast<SCOTCH_Num*>(edgeloctab.data()), nullptr, nullptr))
    {
      throw std::runtime_error("Error building SCOTCH graph");
    }

    SCOTCH_Strat strat;
    SCOTCH_stratInit(&strat);

    // Set SCOTCH strategy
    int nparts = nnodes;
    SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATQUALITY, nparts, nparts,
                               0.0);

    std::vector<SCOTCH_Num> node_partition(mpi_size);

    SCOTCH_randomReset();

    // Partition graph
    if (SCOTCH_dgraphPart(&dgrafdat, nparts, &strat, node_partition.data()))
      throw std::runtime_error("Error during SCOTCH partitioning");

    std::cout << "node partition = ";
    for (auto q : node_partition)
      std::cout << q << ", ";
    std::cout << "\n";

    int c = 0;
    for (int i = 0; i < nparts; ++i)
    {
      for (int j = 0; j < mpi_size; ++j)
      {
        if (node_partition[j] == i)
        {
          renumbering[j] = c;
          ++c;
        }
      }
    }

    std::cout << "renumbering = ";
    for (auto q : renumbering)
      std::cout << q << ", ";
    std::cout << "\n";
  }

  MPI::broadcast(mpi_comm, renumbering);

  // Renumber
  for (auto& q : _dest_processes)
    q = renumbering[q];
}
