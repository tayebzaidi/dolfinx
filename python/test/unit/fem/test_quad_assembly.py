# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest

from dolfinx import FunctionSpace, fem, UnitSquareMesh, UnitCubeMesh, Mesh
from dolfinx.fem import assemble_vector
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.cpp.mesh import CellType, GhostMode
from ufl import TestFunction, div, dx


def get_mesh(cell_type, datadir):
    if MPI.COMM_WORLD.size == 1:
        if cell_type == CellType.triangle:
            return UnitSquareMesh(MPI.COMM_WORLD, 2, 1, cell_type)
        elif cell_type == CellType.quadrilateral:
            points = np.array([[0., 0.], [0.5, 0.], [1., 0.],
                               [0., .5], [0.5, .5], [1., .5],
                               [0., 1.], [0.5, 1.], [1., 1.]])
            cells = [[0, 1, 3, 4], [4, 1, 5, 2], [3, 4, 6, 7], [4, 5, 7, 8]]
            mesh = Mesh(MPI.COMM_WORLD, cell_type, points, cells,
                        [], GhostMode.none)
            mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
            mesh.create_connectivity_all()
            return mesh
        else:
            return UnitCubeMesh(MPI.COMM_WORLD, 2, 1, 1, cell_type)
    else:
        if cell_type == CellType.triangle:
            filename = "UnitSquareMesh_triangle.xdmf"
        elif cell_type == CellType.quadrilateral:
            filename = "UnitSquareMesh_quad.xdmf"
        elif cell_type == CellType.tetrahedron:
            filename = "UnitCubeMesh_tetra.xdmf"
        elif cell_type == CellType.hexahedron:
            filename = "UnitCubeMesh_hexahedron.xdmf"
        with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
            return xdmf.read_mesh(name="Grid")


# Run tests on all spaces in periodic table on triangles and tetrahedra
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1])
def test_P_simplex(family, degree, datadir):
    mesh = get_mesh(CellType.quadrilateral, datadir)
    V = FunctionSpace(mesh, (family, degree))

    v = TestFunction(V)
    a = div(v) * dx
    assemble_vector(a)
