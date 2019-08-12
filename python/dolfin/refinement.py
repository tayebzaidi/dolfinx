# -*- coding: utf-8 -*-
# Copyright (C) 2019 Chris N. Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Mesh refinement"""

from dolfin import cpp

def refine(mesh, marker=None, redistribute=True):
    """Refine a mesh by edge bisection

       Parameters
       ----------
       mesh
           The Mesh to be refined
       marker
           A MeshFunction marking entities to be refined
       redistribute
           If True, the new Mesh is repartitioned

       Returns
       -------
       mesh.Mesh
           The new refined Mesh

       """

    if marker is None:
        return cpp.refinement.refine(mesh, redistribute)

    return cpp.refinement.refine(mesh, marker, redistribute)
