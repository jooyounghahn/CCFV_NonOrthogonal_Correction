Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
SPDX-License-Identifier: GPL-3.0-or-later

==============================================================================
Reproduction of the convergence tests of a cell-centered finite volume method
with non-orthogonal correction on polytopal meshes
==============================================================================

This code reproduces the convergence tests of the paper

    Hyeokjoo Park and Jooyoung Hahn,
    "Stability and error analysis of cell-centered finite volume methods with
    non-orthogonal correction on polytopal meshes"

WHAT IS SOLVED

Both examples solve the Poisson problem

    -Laplace(phi) = f  in Omega,      phi = g  on the boundary of Omega,

where f and g are computed from a known exact solution phi:

    Example 1 (2D)   Omega = (-1.25, 1.25)^2,
                     phi = sin^2 x + cos^2 y + ((x + y)/sqrt(2))^3
    Example 2 (3D)   Omega = (-1.25, 1.25)^3,
                     phi = sin^2 x + cos^2 y + z^3

Each example is solved on four meshes N = 1, ..., 4 that are refined
uniformly: polygonal meshes in 2D (dual of a Delaunay triangulation) and
polyhedral meshes in 3D (dual of a tetrahedral mesh). These meshes are not
orthogonal, so the scheme needs a non-orthogonal correction.

The scheme is the cell-centered finite volume method of the paper: a two-point
flux with an over-relaxed non-orthogonal correction, a weighted least-squares
gradient in every cell, and a weighted average of the two cell gradients on
every interior facet. Other choices of the correction and of the weights can
be selected in settings.txt (section 2).

WHAT THE CODE DOES

  1. mesh/make_mesh_2d.py and mesh/make_mesh_3d.py generate the four meshes
     of each example, the same meshes as in the paper.
  2. convergence/run_example1_2d.py (Example 1) and convergence/run_example2_3d.py
     (Example 2) solve the problem on the four meshes, compute the errors of
     the discrete solution in a discrete H^1 seminorm and in the L^2 norm and
     the experimental convergence orders, and print the convergence table.
     With the setting of the paper, they compare the table with the paper
     and print PASS or FAIL.

This is a verification aid for the paper, not a software package. It ships no
meshes and no results: you generate both with the commands below.

NUMBERING IN THE PAPER

Equation, table and test numbers of the paper can change in a revision, so
the code and this file do not use them. In the version of the paper used
here, Example 1 is "Test 2" with its errors in Table 4.3 and its meshes in
Table 4.1, and Example 2 is "Test 3" with Table 4.6 and Table 4.4.


1. QUICK START
==============

You need x86_64 Linux with glibc 2.27 or newer (for the numpy and gmsh wheels)
and bash, OpenFOAM v2606 (openfoam.com) with scotch and MPI, Python 3.12, the
system libraries that the gmsh wheel needs, and internet access for pip (it
downloads about 80 MB). The tested installation is described in section 7,
"Installation on the tested system".

(0) Environment
---------------

Get the repository; every later command runs in its directory.

    git clone https://github.com/jooyounghahn/CCFV_NonOrthogonal_Correction.git && cd CCFV_NonOrthogonal_Correction

Once, in the repository directory: load Intel MPI and OpenFOAM v2606, create
the Python environment and build the solver. Change the two installation paths
if yours differ.

    source /opt/intel/oneapi/mpi/2021.10.0/env/vars.sh            # Intel MPI 2021.10
    export I_MPI_FABRICS=shm                                       # one machine: shared memory
    source ~/OpenFOAM/OpenFOAM-v2606/etc/bashrc WM_MPLIB=INTELMPI  # OpenFOAM v2606
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt                                # gmsh 4.15.2, numpy, matplotlib
    solver/Allwmake                                                # builds bin/ccfvPoisson, about 11 s

In every new shell afterwards, in the repository directory:

    source /opt/intel/oneapi/mpi/2021.10.0/env/vars.sh
    export I_MPI_FABRICS=shm
    source ~/OpenFOAM/OpenFOAM-v2606/etc/bashrc WM_MPLIB=INTELMPI
    source .venv/bin/activate

If your OpenFOAM was built with another MPI library, for example the default
system Open MPI of a standard source build, replace the first three lines of
both blocks by "source ~/OpenFOAM/OpenFOAM-v2606/etc/bashrc" (no Intel MPI
lines and no WM_MPLIB=INTELMPI argument). This setup was not tested.

(1) Meshes
----------

There is one command per test, and each uses one core.

    python mesh/make_mesh_2d.py          # Example 1: meshes/example1/N1 ... N4, about 2 s
    python mesh/make_mesh_3d.py          # Example 2: meshes/example2/N1 ... N4, about 70 s

Each script prints one line per mesh. The cell counts are the numbers |P_h| of
cells in the mesh tables of the paper:

    example1/N1: 273 cells, 1366 faces, 1096 points; check_mesh OK; 0.2 s
    example1/N2: 985 cells, 4926 faces, 3944 points; check_mesh OK; 0.3 s
    example1/N3: 3741 cells, 18706 faces, 14968 points; check_mesh OK; 0.5 s
    example1/N4: 14891 cells, 74456 faces, 59568 points; check_mesh OK; 1.2 s
    example2/N1: 2487 cells, 15893 faces, 13937 points; check_mesh OK; 0.8 s
    example2/N2: 15249 cells, 101052 faces, 86654 points; check_mesh OK; 4.3 s
    example2/N3: 46752 cells, 314343 faces, 268809 points; check_mesh OK; 15.9 s
    example2/N4: 105766 cells, 716856 faces, 612830 points; check_mesh OK; 47.5 s

In 3D the face count is the number |F_h| of facets in the mesh table. In 2D the count includes the front
and back faces of the one-cell-thick slab (see section 4), so
|F_h| = faces - 2 cells.

(2) Convergence tests
---------------------

By default the solver runs with 4 MPI processes (np in settings.txt).

    python convergence/run_example1_2d.py   # Example 1, about 2 s
    python convergence/run_example2_3d.py   # Example 2, about 13 s

Add "--np 1" for a serial run. It gives the same tables.

For each mesh, a runner prints one progress line. It then prints the table,
the table rows as they appear in the LaTeX source of the paper, the values in
full precision, the numbers of cells and facets of the meshes next to |P_h|
and |F_h| in the mesh tables of the paper, and the comparison with the
paper. The expected output contains ("..." marks omitted lines):

    Example 1: errors and experimental convergence orders.
    ---------------------------------------------------------------------
                  |phi_I^* - phi_h|_h       ||phi_I^* - phi_h||_{0,Omega}
       h          Error          Order           Error          Order
    ---------------------------------------------------------------------
     0.282      1.199e-01         ---          1.986e-02         ---
     0.143      5.035e-02        1.272         5.224e-03        1.958
     0.077      2.331e-02        1.249         1.321e-03        2.229
     0.039      1.048e-02        1.173         3.379e-04        2.002
    ---------------------------------------------------------------------
    ...
    PASS: the table is identical to the table of Example 1 in the paper.

    Example 2: errors and experimental convergence orders.
    ---------------------------------------------------------------------
                  |phi_I^* - phi_h|_h       ||phi_I^* - phi_h||_{0,Omega}
       h          Error          Order           Error          Order
    ---------------------------------------------------------------------
     0.498      2.897e-01         ---          6.266e-02         ---
     0.249      1.233e-01        1.231         1.785e-02        1.809
     0.183      7.384e-02        1.674         8.285e-03        2.507
     0.132      5.138e-02        1.118         4.749e-03        1.716
    ---------------------------------------------------------------------
    ...
    PASS: the table is identical to the table of Example 2 in the paper.

The runners also write the table and the comparison to results/example1/table.txt
and results/example2/table.txt. A runner exits with status 0 on PASS. It exits
with 1 on FAIL or when a run fails or does not converge. It exits with 2 when
something is missing (the OpenFOAM or MPI environment, the solver or the
meshes, or a mesh is incomplete); the message says what is missing and, for
the solver and the meshes, the command to run. Load the environment as in step
(0). When the meshes differ from those of the paper (another gmsh version, for
example), the runner says so before the comparison. The times above are
wall-clock times on the tested machine (AMD Ryzen 7 9700X).


2. SETTINGS
===========

settings.txt selects the scheme and the number of MPI processes for both
runners. The defaults are the setting of the convergence tables of the paper:

    correction = over-relaxed
    ls_weights = iii
    average_weights = iii
    np = 4

    key              values ([paper])                   meaning
    ---------------  ---------------------------------  ---------------------------------
    correction       minimum, full-area, [over-relaxed] non-orthogonal correction c_F
                                                        of the numerical flux
    ls_weights       i, ii, [iii]                       weights omega_{K,F} of the
                                                        least-squares gradient
    average_weights  i, ii, [iii]                       weights alpha_{K,F}, alpha_{L,F}
                                                        of the facet average
    np               [4], any positive integer          number of MPI processes;
                                                        1 runs the solver serially

In the notation of the paper (the symbols are defined in settings.txt), the
flux splits the facet normal as n_{K,F} = c_F r_{K,F} + e_{K,F} and

    c_F = cos(theta_F)                                    minimum
          1                                               full-area
          1/cos(theta_F)                                  over-relaxed

    omega_{K,F} = 1                                       (i)
                  |F|                                     (ii)
                  alpha_{K,F} d_F |F| / (|K| cos^2(theta_F))   (iii)

    {{v_h}}_F = alpha_{K,F} v_h|_K + alpha_{L,F} v_h|_L, with
    (alpha_{K,F}, alpha_{L,F}) = (1/2, 1/2)                   (i)
                                 (d_{L,F}/d_F, d_{K,F}/d_F)   (ii)
                                 (d_{K,F}/d_F, d_{L,F}/d_F)   (iii)

The facet average is taken on an interior facet F between the cells K and L.
On a boundary facet alpha_{K,F} = 1, so alpha_{K,F} d_F = d_{K,F} there. The
least-squares weight (iii) contains alpha_{K,F} and therefore follows the
choice of average_weights. settings.txt repeats these formulas in its
comments.

To change a setting, edit settings.txt (one "key = value" per line; "#" starts
a comment), or pass another file with "--settings FILE". The option "--np N"
overrides np. With the setting of the paper, the runners compare the table
with the paper for any np. With any other setting they print the table and the
line "No comparison with the table of Example 1 (or 2) is made", and they exit with status 0 if
all runs converged. The runners have two more options: "--meshes DIR" (default
meshes/) and "--out DIR" (default results/). Each run replaces
results/<test>/N<k>/np<np>/ and results/<test>/table.txt. To keep the results
of another setting, give it its own "--out DIR".


3. RESOURCES
============

All values were measured on the tested machine: AMD Ryzen 7 9700X (8 cores,
16 threads), 123 GiB RAM, Red Hat Enterprise Linux 8.10.

    command                               cores  wall    peak memory,   largest process        recommended
                                                 time    all processes                         free memory
    ------------------------------------  -----  ------  -------------  ---------------------  -----------
    solver/Allwmake                       1      10.3 s  466 MiB        466 MiB (g++)          1 GiB
    mesh/make_mesh_2d.py                  1      2.3 s   100 MiB        96 MiB                 0.25 GiB
    convergence/run_example1_2d.py (np = 4)  4      2.2 s   197 MiB        89 MiB                 0.25 GiB
    convergence/run_example1_2d.py --np 1    1      0.8 s   141 MiB        81 MiB                 0.25 GiB
    mesh/make_mesh_3d.py                  1      68 s    600 MiB        590 MiB (polyDualMesh, 1 GiB
                                                                        N4)
    convergence/run_example2_3d.py (np = 4)  4      12.9 s  658 MiB        452 MiB (runner,       1 GiB
                                                                        h of N4)
    convergence/run_example2_3d.py --np 1    1      11.1 s  584 MiB        453 MiB (runner,       1 GiB
                                                                        h of N4)

Memory. "Largest process" is the peak resident set size (RSS) of a single
  process, measured with GNU "time -v". "Peak memory, all processes" is the
  largest sum of the proportional set sizes (PSS) of all processes that ran at
  the same time. PSS counts shared libraries and MPI shared memory once, so the
  sum is the memory actually used. Summing the RSS of the four MPI ranks would
  count them four times. The peak memory of all processes is the least free
  memory a command needs; the recommended free memory adds a margin to it and
  is rounded up. Every step runs in 1 GiB of free memory. A parallel build
  "solver/Allwmake -j N" needs about N x 0.5 GiB.

CPU. Mesh generation uses one core. gmsh runs with one thread, the OpenFOAM
  utilities run serially, and no MPI is involved. By default the tests use MPI
  with 4 processes: for each mesh, the runner decomposes the mesh with
  decomposePar (method scotch, np subdomains) and runs
  "mpirun -np 4 bin/ccfvPoisson -parallel". With np = 1 the solver runs
  serially without mpirun.

np and the paper. Serial runs and 4-process runs give the same errors to 11
  significant digits, which is every digit the solver prints, so the tables do
  not depend on np (np = 1, 2 and 4 were tested). The number of outer
  iterations is also the same. Only the final residual (by less than 0.1 %)
  and the number of linear-solver iterations differ.

Disk. The meshes take about 120 MiB (Example 1: 9 MiB, Example 2: 112 MiB). The
  results take about 135 MiB with np = 4, most of it for the decomposed 3D
  meshes, and less than 1 MiB with np = 1. The Python environment .venv takes
  about 280 MiB.


4. WHAT THE SCRIPTS DO: MESHES (mesh/)
======================================

make_mesh_2d.py and make_mesh_3d.py write level N as an OpenFOAM case
directory meshes/example1/N<N> or meshes/example2/N<N>. The directory holds
constant/polyMesh (ASCII, 17 significant digits, which store every point
coordinate exactly), system/, logs/ with the log, wall time and peak memory of
every step, and MESH.md, which describes the mesh and its construction.
Options: "--out DIR" (another mesh root), "--levels 1 2" (only some levels) and
"--figures" (also draw figures, which needs matplotlib). The scripts never
overwrite an existing level directory: they stop with a message. Remove the
directory to make that level again.

A level is built in N<N>.partial and renamed to N<N> after its last step, so
N<N> always holds a complete mesh. If a step fails or the script is
interrupted, the script prints the end of the failing log and the command that
makes the remaining levels. N<N>.partial stays for inspection, and the next
run replaces it. To draw the figures of an existing mesh, run
"python mesh/plot_mesh.py meshes/example2/N1". The figures go to figures/ in the
level directory.

Example 1 (make_mesh_2d.py), target edge length lc = 0.19/2^(N-1), that is
0.19, 0.095, 0.0475, 0.02375:

  1. gmsh: a Delaunay triangulation (Mesh.Algorithm 5) of (-1.25, 1.25)^2 with
     the uniform target edge length lc (Mesh.MeshSizeMin = Mesh.MeshSizeMax =
     lc, Mesh.Optimize 1).
  2. The dual mesh (Python, numpy). It has one polygonal cell per vertex of the
     triangulation. The cell vertices are the centroids of the surrounding
     triangles and, at the boundary, the midpoints of the boundary edges and
     the corners of the square. A cell therefore has a single boundary facet on
     each side it touches. OpenFOAM stores the mesh as a slab one cell thick
     (0 <= z <= lc) with the patches "boundary" (the four sides) and
     "frontAndBack" (type empty).
  3. OpenFOAM "renumberMesh -overwrite".
  4. check_mesh.py (see below).

Example 2 (make_mesh_3d.py), target edge length lc = 0.19/N, that is 0.19,
0.095, 0.0633333333333333, 0.0475:

  1. gmsh: a tetrahedral mesh of (-1.25, 1.25)^3 with the uniform target edge
     length lc. It uses Frontal-Delaunay triangles on the sides
     (Mesh.Algorithm 6), Delaunay tetrahedra (Mesh.Algorithm3D 1),
     Mesh.Optimize 1 and Mesh.OptimizeNetgen 1, and writes MSH 2.2 ASCII.
  2. OpenFOAM "gmshToFoam".
  3. OpenFOAM "polyDualMesh 60 -overwrite": the dual polyhedral mesh, with one
     cell per vertex of the tetrahedral mesh. The edges and corners of the box
     are kept as features (feature angle 60 degrees).
  4. OpenFOAM "combinePatchFaces 1 -overwrite": the boundary faces of a cell on
     one side of the box are merged into one face.
  5. OpenFOAM "renumberMesh -overwrite".
  6. check_mesh.py.

check_mesh.py checks four rules:

  - each corner of the domain lies in exactly one cell;
  - a cell that touches an edge of the domain touches it along one of its own
    edges, never in a single vertex;
  - each domain edge is split into segments without gaps, one cell per
    segment, and neighbouring cells on the edge share a face;
  - a cell has at most one boundary face on each side of the domain.

If a rule fails, the mesh is moved to constant/polyMesh.rejected in
N<N>.partial and the script stops with an error. The paper meshes pass all
four rules.

Why 0.0633333333333333. The meshes of the paper were made with the target edge
lengths written as these decimal strings, and the scripts pass the same
strings to gmsh. For N = 3 the string is 0.19/3 rounded to 15 significant
digits. Computing 0.19/3 in double precision instead gives a number that is
3 units in the last place larger. gmsh 4.15.2 then makes a different mesh,
with 46814 instead of 46752 cells, and the table of Example 2 changes. In Example 1 there is
no such difference: the strings for 0.19/2^(N-1) parse to exactly the
computed values.

Why one thread. gmsh runs with General.NumThreads = 1. Its result depends on
the number of threads: with 4 threads, gmsh 4.15.2 makes different 3D meshes,
for example 15266 instead of 15249 cells for N = 2. With one thread, every run
gives the same mesh. gmsh also starts without reading user configuration
files.


5. WHAT THE SCRIPTS DO: CONVERGENCE TESTS (convergence/) AND SOLVER (solver/)
=============================================================================

Convergence tests
-----------------

run_example1_2d.py and run_example2_3d.py share common.py. First they check that
every level is a complete mesh of the mesh script: constant/polyMesh and
MESH.md, which the script writes after its last step. Then they do the
following for each level N = 1, ..., 4:

  1. Write the case results/<test>/N<N>/np<np>/. Its constant/polyMesh is a
     symbolic link to the mesh, which is only read. Its system/ holds the
     dictionaries of convergence/case/system, the solver dictionary ccfvDict
     written from settings.txt and, for np > 1, a decomposeParDict (scotch).
     A runner replaces only case directories that a runner wrote (marker file
     RUN.txt).
  2. Run decomposePar and "mpirun -np <np> bin/ccfvPoisson -parallel" for
     np > 1, or "bin/ccfvPoisson" for np = 1. Read the solver's RESULT line
     and check that the run converged.
  3. Compute the mesh size h_N = max_K h_K, where h_K is the largest distance
     between two vertices of the cell K (in 2D only x and y are used), and
     count the cells and the facets (in 2D without the front and back faces of
     the slab). The counts are compared with |P_h| and |F_h| in the mesh tables
     of the paper; a difference means that the meshes are not those of the
     paper, and the runner reports it.

The experimental order between the meshes N-1 and N is
log(E_N/E_{N-1}) / log(h_N/h_{N-1}). It is computed from h_N in double
precision and from the errors as the solver prints them (11 significant
digits). Only the printed table is rounded, as in the paper: h to 3 decimals,
errors to 4 significant digits, orders to 3 decimals. The result is PASS when
every row equals the row of the paper in convergence/expected/ and every run
converged. The rows in convergence/expected/ were copied from the LaTeX source
of the paper.

Solver
------

bin/ccfvPoisson, built from solver/ccfvPoisson by solver/Allwmake
(solver/Allwclean removes the build), is an OpenFOAM v2606 application. It
solves -Laplace(phi) = f with Dirichlet data by the cell-centered finite
volume method of the paper. The method uses the numerical flux with the
non-orthogonal correction, the weighted least-squares gradient with the
weights omega_{K,F}, and the facet average with the weights alpha_{K,F}
(section 2 and settings.txt).

  - One program handles the 2D slabs and the 3D meshes. On a slab, |K| is the
    cell area and |F| the edge length.
  - As in the numerical tests of the paper, the facet and cell quantities are those of
    OpenFOAM, the right-hand side is |K| f(x_K), and the boundary values are
    phi(x_F).
  - The system is solved by deferred correction, starting from phi^0 = 0.
    Step k solves A phi^k = b(phi^{k-1}), where A is the symmetric two-point
    matrix and b holds the right-hand side, the boundary terms and the
    correction terms with the gradient of phi^{k-1}.
  - Each linear system is solved by conjugate gradients with an algebraic
    multigrid preconditioner (system/fvSolution, absolute tolerance 1e-15).
  - The iteration stops when the residual of the discrete problem,
    res = sum_K |(A phi^k - b(phi^k))_K| / sum_K |A_KK|,
    is below epsilon = 1e-12, after at most 200 steps.

The solver reads system/ccfvDict:

    solution        example1;     // example1 | example2 | linear
    correction      overRelaxed;  // minimum | fullArea | overRelaxed   (correction)
    lsWeights       iii;          // i | ii | iii                       (least-squares weights)
    averageWeights  iii;          // i | ii | iii                       (facet average weights)
    epsilon         1e-12;        // stopping tolerance of the deferred correction
    maxOuter        200;          // largest number of steps
    writeFields     false;        // true: write phi_h to the time directory 1

It prints one line that starts with RESULT. Among other values, the line
contains E2 = ||phi_I^* - phi_h||_{0,Omega} and H1h = |phi_I^* - phi_h|_h
(with phi_I^*|_K = phi(x_K)), the number of steps K and the final residual
res. All floating-point values are printed in the format %.10e.

"solution linear" (phi = 1 + 2x - 3y in 2D, 1 + 2x - 3y + 4z in 3D, f = 0)
checks the implementation. The scheme is exact for affine functions with every
correction and every choice of weights, so the errors come only from the
stopping tolerance and round-off: with epsilon = 1e-12, E2, H1h and Linf were
below 1e-10 in 2D and below 2e-9 in 3D on the meshes N = 1, for all 27
combinations of correction and weights (np = 1). Do not set epsilon below
about 1e-14: the residual does not fall below round-off (about 1e-15).

To run the solver by hand, edit system/ccfvDict in a case directory written by
a runner. Then run "bin/ccfvPoisson -case <case>" (np = 1) or
"mpirun -np <np> bin/ccfvPoisson -parallel -case <case>" (a case written for
that np).


6. REPOSITORY LAYOUT
====================

    settings.txt                  scheme and number of MPI processes of the convergence tests
    requirements.txt              Python packages (gmsh 4.15.2, numpy 2.5.3, matplotlib 3.9.2)
    LICENSE                       GNU General Public License, version 3
    mesh/make_mesh_2d.py          Example 1 meshes -> meshes/example1/N1 ... N4
    mesh/make_mesh_3d.py          Example 2 meshes -> meshes/example2/N1 ... N4
    mesh/check_mesh.py            corner, edge and side rules (run by the two scripts above)
    mesh/plot_mesh.py             figures of a mesh (optional, needs matplotlib)
    mesh/system/                  dictionaries read by the OpenFOAM mesh utilities
    solver/Allwmake, Allwclean    build or remove bin/ccfvPoisson (wmake)
    solver/ccfvPoisson/           solver source: ccfvPoisson.C (main program), ccfvScheme (settings),
                                  ccfvGeometry (facet and cell quantities), ccfvGradient
                                  (least-squares gradient), ccfvFlux (flux, matrix, deferred
                                  correction), ccfvNorms (errors), ccfvExactSolution (test problems),
                                  ccfvHalo (parallel exchange), ccfvFormat (number format)
    convergence/run_example1_2d.py   Example 1 -> its convergence table
    convergence/run_example2_3d.py   Example 2 -> its convergence table
    convergence/common.py         code shared by the two runners
    convergence/case/system/      dictionaries copied into every case
    convergence/expected/         rows of the two convergence tables from the LaTeX source of the paper

Generated, and ignored by git: bin/, meshes/, results/, .venv/ and the build
directory solver/ccfvPoisson/Make/<platform>/. To make the meshes and the
results again, remove them first: "rm -rf meshes results" (the mesh scripts
never overwrite an existing level).


7. INSTALLATION ON THE TESTED SYSTEM
====================================

    OS                  Red Hat Enterprise Linux 8.10 (Ootpa),
                        kernel 4.18.0-553.81.1.el8_10.x86_64, glibc 2.28
    CPU, memory         AMD Ryzen 7 9700X, 8 cores / 16 threads, 1 socket; 123 GiB
    C/C++ compiler      system GCC 8.5.0 20210514 (Red Hat 8.5.0-28),
                        rpm gcc and gcc-c++ 8.5.0-28.el8_10
    other build tools   GNU make 4.2.1, flex 2.6.1, zlib-devel 1.2.11
    OpenFOAM            v2606 (openfoam.com), api 2606, patch 0,
                        build _481094fdf3-20260618, built from source
    OpenFOAM options    WM_OPTIONS=linux64GccDPInt32Opt (system Gcc, double precision,
                        32-bit labels, optimised), WM_MPLIB=INTELMPI
    solver compilation  wmake defaults: g++ -std=c++17 -m64 -pthread ... -O3 ... -fPIC
                        (no -march, no -ffast-math), linked with -lfiniteVolume -lOpenFOAM
    MPI                 Intel MPI Library 2021.10, Build 20230619 (Intel oneAPI, installed
                        with the Intel installer under /opt/intel/oneapi), I_MPI_FABRICS=shm
    scotch              6.1.0 from ThirdParty-v2606 (used by decomposePar)
    Python              CPython 3.12 as the base of the venv (python3.12 -m venv):
                        verified with 3.12.7 (Anaconda build, pip 24.2) and with 3.12.13
                        (RHEL 8 package python3.12-3.12.13-2.el8_10, pip 23.2.1); both give
                        the same meshes byte for byte and the same tables
    Python packages     gmsh 4.15.2 (wheel py2.py3-none-manylinux_2_24_x86_64), numpy 2.5.3,
                        matplotlib 3.9.2, and the matplotlib dependencies contourpy 1.4.0,
                        cycler 0.12.1, fonttools 4.68.0, kiwisolver 1.5.1, packaging 26.3,
                        pillow 13.0.0, pyparsing 3.3.3, python-dateutil 4.0.0, tzdata 2026.6
    system libraries    mesa-libGLU 9.0.0, libglvnd-glx 1.3.4, libX11 1.6.8, libXext 1.3.4,
    of the gmsh wheel   libXrender 0.9.10, libXcursor 1.1.15, libXfixes 5.0.3, libXft 2.3.3,
                        libXinerama 1.1.4, fontconfig 2.13.1

Intel MPI. On the tested system, Intel MPI 2021.10 was installed with the
Intel oneAPI installer under /opt/intel/oneapi. Its environment comes from
/opt/intel/oneapi/mpi/2021.10.0/env/vars.sh. On one machine,
"export I_MPI_FABRICS=shm" selects the shared-memory transport.

OpenFOAM v2606. Build it from the source packs OpenFOAM-v2606.tgz and
ThirdParty-v2606.tar.gz (https://dl.openfoam.com/source/v2606/), unpacked side
by side in ~/OpenFOAM. The prerequisites are listed in
OpenFOAM-v2606/doc/Requirements.md.

    source /opt/intel/oneapi/mpi/2021.10.0/env/vars.sh
    export I_MPI_FABRICS=shm
    source ~/OpenFOAM/OpenFOAM-v2606/etc/bashrc WM_MPLIB=INTELMPI
    cd ~/OpenFOAM/OpenFOAM-v2606
    ./Allwmake -j -s -q -l        # also builds scotch 6.1.0 of ThirdParty-v2606;
                                  # log in log.linux64GccDPInt32Opt

Every other setting is the OpenFOAM default: system GCC, double precision,
32-bit labels and an optimised build. The mesh generators use the OpenFOAM
utilities gmshToFoam, polyDualMesh, combinePatchFaces and renumberMesh. The
parallel runs use decomposePar with scotch and mpirun.

Python. On RHEL 8, "sudo dnf install python3.12" provides python3.12; its venv
module puts pip into the new environment. The gmsh wheel links against X11 and
OpenGL libraries even though it opens no window. Without them, "import gmsh"
fails, and the mesh scripts stop with the import error. On RHEL 8, install
them with:

    sudo dnf install mesa-libGLU libglvnd-glx libX11 libXext libXrender libXcursor libXfixes libXft libXinerama fontconfig

dnf installs the remaining libraries as dependencies. Package names on other
distributions were not checked. Then create the environment in the repository
directory:

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Reproducibility. On this system, make_mesh_2d.py and make_mesh_3d.py reproduce
the meshes of the paper byte for byte: every file in constant/polyMesh of all
eight meshes. The values K, res, E2, H1h, Linf and linIter printed by the
solver equal those of the runs made for the paper in every printed digit, for
np = 1 and np = 4. This was verified only with the versions above (two Python
3.12 builds, see the table). The output of gmsh depends on its version and on
the number of threads, and it may also depend on the system math library and
on the CPU. With other versions the meshes can differ, and the tables are then
likely to differ in the last printed digits. Another OpenFOAM version,
compiler or MPI library may also change the last digits. With np > 1, the
scotch decomposition and the MPI library can change res and the iteration
counts.


8. LICENSE
==========

Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn

This code is free to download, use, copy, modify and distribute under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version (SPDX-License-Identifier: GPL-3.0-or-later). Modified versions that
are distributed must be released under the same license. The code is
distributed WITHOUT ANY WARRANTY. See the file LICENSE for the full text.


9. HOW TO CITE
==============

If you use this code, please cite the paper:

    Hyeokjoo Park and Jooyoung Hahn, "Stability and error analysis of
    cell-centered finite volume methods with non-orthogonal correction on
    polytopal meshes".

    @misc{ccfv_nonorthogonal_correction,
      author = {Hyeokjoo Park and Jooyoung Hahn},
      title  = {Stability and error analysis of cell-centered finite volume methods with
                non-orthogonal correction on polytopal meshes}
    }
