// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PartitionData.h"
#include <algorithm>
#include <sstream>

extern "C"
{
#include <ptscotch.h>
}

using namespace dolfin;
using namespace dolfin::mesh;

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
int PartitionData::num_ghosts() const
{
  return _offset.size() - _dest_processes.size() - 1;
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
    std::vector<std::set<int>> edgeconn(mpi_size);

    for (std::size_t i = 0; i < recv_data.size(); i += 3)
    {
      edgeconn[recv_data[i]].insert(recv_data[i + 1]);
      std::pair<int, int> idx(recv_data[i], recv_data[i + 1]);
      neighbour_info[idx] += recv_data[i + 2];
    }

    std::vector<SCOTCH_Num> vertloctab = {0};
    std::vector<SCOTCH_Num> edgeloctab;
    for (auto& node : edgeconn)
    {
      edgeloctab.insert(edgeloctab.end(), node.begin(), node.end());
      vertloctab.push_back(edgeloctab.size());
    }
    assert((int)vertloctab.size() == mpi_size + 1);
    const SCOTCH_Num vertlocnbr = mpi_size;

    std::vector<SCOTCH_Num> edloloctab(edgeloctab.size(), 0);
    for (std::size_t i = 0; i < recv_data.size(); i += 3)
    {
      int from = recv_data[i];
      int to = recv_data[i + 1];
      int weight = recv_data[i + 2];
      int pos = std::find(edgeloctab.begin() + vertloctab[from],
                          edgeloctab.begin() + vertloctab[from + 1], to)
                - edgeloctab.begin();
      edloloctab[pos] += weight;
    }

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
//-----------------------------------------------------------------------------
void PartitionData::optimise(MPI_Comm mpi_comm)
{
  int rank = MPI::rank(mpi_comm);
  int size = MPI::size(mpi_comm);

  // Find most common process number in local partition and 'claim' it.

  std::vector<int> count(size, 0);
  for (const auto& dp : _dest_processes)
    ++count[dp];

  std::stringstream s;
  s << rank << " = [";
  for (auto& q : count)
    s << q << " ";
  s << "]\n";

  std::cout << s.str();

  std::vector<int> favoured_partition
      = {(int)(std::max_element(count.begin(), count.end()) - count.begin())};

  std::cout << "Process " << rank << " would like to claim partition "
            << favoured_partition[0] << "\n";

  // Gather up remapping information
  std::vector<int> process_to_partition;
  MPI::all_gather(mpi_comm, favoured_partition, process_to_partition);

  // Check for uniqueness
  std::fill(count.begin(), count.end(), 0);
  for (const auto& q : process_to_partition)
    ++count[q];
  int max_count = *std::max_element(count.begin(), count.end());
  int min_count = *std::min_element(count.begin(), count.end());

  if (max_count == 1 and min_count == 1)
  {
    std::cout << "Uniqueness OK\n";

    // Invert map
    std::vector<int> partition_to_process(size);
    for (std::size_t i = 0; i < process_to_partition.size(); ++i)
      partition_to_process[process_to_partition[i]] = i;

    // Remap partition accordingly
    for (auto& q : _dest_processes)
      q = partition_to_process[q];

    // Check again
    std::fill(count.begin(), count.end(), 0);
    for (const auto& dp : _dest_processes)
      ++count[dp];

    s.str("New: ");
    s << rank << " = [";
    for (auto& q : count)
      s << q << " ";
    s << "]\n";

    std::cout << s.str();
  }
}
