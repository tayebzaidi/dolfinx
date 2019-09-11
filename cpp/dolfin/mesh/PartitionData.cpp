// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PartitionData.h"
#include <algorithm>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
PartitionData::PartitionData(
    const std::vector<int>& cell_partition,
    const std::map<std::int64_t, std::vector<int>>& ghost_procs)
    : _offset(1)

{
  for (std::size_t i = 0; i != cell_partition.size(); ++i)
  {
    const auto it = ghost_procs.find(i);
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

  neighbour_info.clear();
  for (std::size_t i = 0; i < recv_data.size(); i += 3)
  {
    std::pair<int, int> idx(recv_data[i], recv_data[i + 1]);
    neighbour_info[idx] += recv_data[i + 2];
  }

  std::stringstream s;

  for (auto& info : neighbour_info)
  {
    s << info.first.first << ":" << info.first.second << " = " << info.second
      << ",";
  }
  std::cout << s.str() << "\n";
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
