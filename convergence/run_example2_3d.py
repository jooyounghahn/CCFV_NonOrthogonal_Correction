#!/usr/bin/env python3
# Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
"""Example 2: the 3D convergence test of the paper and its convergence table.

Solves -Laplace(phi) = f on (-1.25, 1.25)^3 with
phi = sin^2 x + cos^2 y + z^3 and Dirichlet data phi(x_F) on the four
polyhedral meshes N = 1, ..., 4 (made by mesh/make_mesh_3d.py), then prints
the errors |phi_I^* - phi_h|_h and ||phi_I^* - phi_h||_{0,Omega} with their
experimental orders as in the convergence table of the paper.  The scheme and the number of
MPI processes are read from settings.txt.

Run inside an OpenFOAM v2606 environment, from the repository directory:

    python3 convergence/run_example2_3d.py [--meshes DIR] [--out DIR]
                                        [--settings FILE] [--np N]

See convergence/common.py for the steps and the exit status.
"""

import os
import sys

sys.dont_write_bytecode = True      # keep the repository free of __pycache__
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

if __name__ == "__main__":
    sys.exit(common.main(common.EXAMPLE2))
