# Copyright (C) 2020 Joseph P. Dean, Igor A. Baratta
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import ufl
import pytest
import numpy as np


def test_locate_dofs_geometrical():
    """Test that locate_dofs_geometrical when passed two function
    spaces returns the correct degrees of freedom in each space.
    """
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 4, 8)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    W = dolfinx.function.FunctionSpace(mesh, P0 * P1)
    V = W.sub(0).collapse()

    dofs = dolfinx.fem.locate_dofs_geometrical(
        (W.sub(0), V), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))

    # Collect dofs from all processes (does not matter that the numbering
    # is local to each process for this test)
    all_dofs = np.vstack(dolfinx.MPI.comm_world.allgather(dofs))

    # Check only one dof pair is returned
    assert len(all_dofs) == 1

    # On process with the dof pair
    if len(dofs) == 1:
        # Check correct dof returned in W
        coords_W = W.tabulate_dof_coordinates()
        assert np.isclose(coords_W[dofs[0][0]], [0, 0, 0]).all()
        # Check correct dof returned in V
        coords_V = V.tabulate_dof_coordinates()
        assert np.isclose(coords_V[dofs[0][1]], [0, 0, 0]).all()


@pytest.mark.skipif(dolfinx.MPI.comm_world.size == 1,
                    reason="This test should only be run in parallel.")
@pytest.mark.parametrize("mesh", [dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 10, 10),
                                  dolfinx.UnitCubeMesh(dolfinx.MPI.comm_world, 5, 5, 5)])
def test_locate_dofs_topological(mesh):
    """Test locate dofs topological when some dofs on this process can only be
    found by other processes (eg. facet not marked).
    """
    comm = dolfinx.MPI.comm_world
    mesh.create_connectivity_all()
    tdim = mesh.topology.dim

    facets_on_boundary = mesh.topology.on_boundary(tdim - 1)
    marked_facets = np.where(facets_on_boundary)[0]

    # Remove marked boundary facets on process 0 whose dofs
    # can be found by other processes.
    if comm.rank == 0:
        cell_facet = mesh.topology.connectivity(tdim, tdim - 1)
        facet_cell = mesh.topology.connectivity(tdim - 1, tdim)

        one_cell_facets = np.diff(facet_cell.offsets()) == 1
        interface_facets = np.where(np.logical_xor(one_cell_facets, facets_on_boundary))[0]

        for facet in interface_facets:
            cell = facet_cell.links(facet)
            facets = cell_facet.links(cell[0])
            for idx in facets:
                facets_on_boundary[idx] = False

    new_marked_facets = np.where(facets_on_boundary)[0]
    assert comm.allreduce(new_marked_facets.size) != comm.allreduce(marked_facets.size)

    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = dolfinx.function.FunctionSpace(mesh, P0 * P1)
    V0 = V.sub(0).collapse()

    dofs = dolfinx.fem.locate_dofs_topological(V0, tdim - 1, marked_facets, False)
    dofs1 = dolfinx.fem.locate_dofs_topological(V0, tdim - 1, new_marked_facets)

    assert np.all(dofs == dofs1)

    dofs2 = dolfinx.fem.locate_dofs_topological((V.sub(1), V0), tdim - 1, marked_facets, False)
    dofs3 = dolfinx.fem.locate_dofs_topological((V.sub(1), V0), tdim - 1, new_marked_facets)

    assert np.all(dofs2 == dofs3)
