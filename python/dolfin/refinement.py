# -*- coding: utf-8 -*-
# Copyright (C) 2018 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Refinement module for Mesh refinement"""

import dolfin.cpp as cpp


def refine(mesh, markers=None, redistribute=True):
    """Refine a mesh, either uniformly, or optionally by marking some entities for refinement.
       Only meshes of triangles or tetrahedra are supported.
    Parameters
    ----------
    mesh
        The mesh to refine.
    markers
        If supplied, refinement will take place on the entities which are marked. Other
        neighbouring entities may also be refined, to maintain topological integrity.
    redistribute
        In parallel, this flag determines whether the mesh should be
        redistributed after refinement.
    Returns
    -------
        The refined mesh.
    """
    if markers:
        new_mesh = cpp.refinement.refine(mesh, markers, redistribute)
    else:
        new_mesh = cpp.refinement.refine(mesh, redistribute)

    new_mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(new_mesh)
    return new_mesh
