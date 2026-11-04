# Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
"""Shared code of the convergence runners run_example1_2d.py and run_example2_3d.py.

For each mesh level N = 1, ..., 4 of a test the runner

1. builds the OpenFOAM case <results>/<test>/N<k>/np<np>/: constant/polyMesh is
   a symbolic link to <meshes>/<test>/N<k>/constant/polyMesh (the mesh is only
   read), system/ holds the dictionaries of convergence/case/system, the
   solver dictionary ccfvDict written from settings.txt and, for np > 1, a
   decomposeParDict (scotch, np subdomains);
2. runs decomposePar (np > 1) and the solver bin/ccfvPoisson, serially for
   np = 1 and with "mpirun -np <np> ... -parallel" otherwise, and reads the
   single line of the solver output that starts with "RESULT ";
3. computes the mesh size h_N = max_K h_K, where h_K is the largest distance
   between two vertices of the cell K.  The 2D meshes are slabs one cell
   thick in z, and there only x and y enter h_K.

The experimental convergence order between the meshes N-1 and N is
log(E_N/E_{N-1}) / log(h_N/h_{N-1}).  It is computed from h_N in double
precision and from the errors E_N as printed by the solver (11 significant
digits); only the printed values are rounded: h to 3 decimals, the errors to
4 significant digits and the orders to 3 decimals, as in the paper.  The
errors are E_N = |phi_I^* - phi_h|_h (RESULT key H1h) and
E_N = ||phi_I^* - phi_h||_{0,Omega} (RESULT key E2).

Before the runs, the runner checks that every level is a complete mesh made
by the mesh script (constant/polyMesh and MESH.md, which the script writes
last).  After the runs, it compares the numbers of cells and facets of the
meshes with |P_h| and |F_h| in the mesh tables of the paper and reports a
difference, the usual reason why a table differs from the paper.

Exit status of the runners: 0 when all runs converged and, for the setting
of the paper, every row equals the paper; 1 when a run fails or does not
converge, or a row differs from the paper; 2 when the setup is incomplete
(settings, OpenFOAM environment, solver binary or meshes).
"""

import os

# One thread per Python process; also inherited by the solver processes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import collections
import datetime
import gzip
import math
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on the installation
    sys.exit("error: the runners need numpy (pip install -r requirements.txt)")


# ------------------------------------------------------------------- paths
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER = os.path.join(REPO, "bin", "ccfvPoisson")
CASE_SYSTEM = os.path.join(REPO, "convergence", "case", "system")
EXPECTED_DIR = os.path.join(REPO, "convergence", "expected")
DEFAULT_SETTINGS = os.path.join(REPO, "settings.txt")
DEFAULT_MESHES = os.path.join(REPO, "meshes")
DEFAULT_RESULTS = os.path.join(REPO, "results")

LEVELS = (1, 2, 3, 4)
EPSILON = "1e-12"   # stopping rule of the deferred correction (ccfvDict)
MAX_OUTER = 200     # largest number of outer iterations (ccfvDict)
CASE_MARK = "RUN.txt"
CASE_MARK_FIRST_LINE = "Convergence-test case written by convergence/common.py"
POLYMESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")
MESH_DONE = "MESH.md"   # written last by mesh/make_mesh_2d.py and make_mesh_3d.py


class SetupError(Exception):
    """The settings, the environment, the solver or the meshes are not ready."""


class RunError(Exception):
    """A tool or the solver failed, or its output is not as expected."""


# ------------------------------------------------------------------- tests
Test = collections.namedtuple(
    "Test", "key name dim table title problem expected mesh_script "
            "mesh_table paper_cells paper_facets")

# paper_cells and paper_facets: |P_h| and |F_h| of the meshes N = 1, ..., 4
# in the mesh tables of the paper.  They are used only to report whether the meshes are those of
# the paper.
EXAMPLE1 = Test(
    key="example1", name="Example 1", dim=2, table="the table of Example 1",
    title="Example 1: errors and experimental convergence orders.",
    problem="2D, Omega = (-1.25, 1.25)^2, "
            "phi = sin^2 x + cos^2 y + ((x + y)/sqrt(2))^3",
    expected="example1.txt", mesh_script="mesh/make_mesh_2d.py",
    mesh_table="the mesh table of Example 1",
    paper_cells=(273, 985, 3741, 14891),
    paper_facets=(820, 2956, 11224, 44674))

EXAMPLE2 = Test(
    key="example2", name="Example 2", dim=3, table="the table of Example 2",
    title="Example 2: errors and experimental convergence orders.",
    problem="3D, Omega = (-1.25, 1.25)^3, phi = sin^2 x + cos^2 y + z^3",
    expected="example2.txt", mesh_script="mesh/make_mesh_3d.py",
    mesh_table="the mesh table of Example 2",
    paper_cells=(2487, 15249, 46752, 105766),
    paper_facets=(15893, 101052, 314343, 716856))


# ---------------------------------------------------------------- settings
SETTING_KEYS = ("correction", "ls_weights", "average_weights", "np")
# settings.txt word -> word of the solver dictionary ccfvDict
CORRECTION_WORDS = collections.OrderedDict(
    [("minimum", "minimum"), ("full-area", "fullArea"),
     ("over-relaxed", "overRelaxed")])
WEIGHT_CHOICES = ("i", "ii", "iii")
PAPER_SETTING = {"correction": "over-relaxed", "ls_weights": "iii",
                 "average_weights": "iii"}

Settings = collections.namedtuple(
    "Settings", "correction ls_weights average_weights nprocs")


def read_settings(path):
    """Read and check settings.txt ("key = value" lines, "#" comments)."""
    if not os.path.isfile(path):
        raise SetupError(f"settings file not found: {path}")
    values = {}
    # utf-8-sig: a byte-order mark written by some editors is skipped
    with open(path, encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            where = f"{path}, line {lineno}"
            if "=" not in line:
                raise SetupError(f"{where}: expected 'key = value', "
                                 f"found {raw.strip()!r}")
            key, value = (s.strip() for s in line.split("=", 1))
            key = key.lower()
            if key not in SETTING_KEYS:
                raise SetupError(f"{where}: unknown key {key!r} "
                                 f"(known keys: {', '.join(SETTING_KEYS)})")
            if key in values:
                raise SetupError(f"{where}: {key} is set twice")
            values[key] = value.lower()
    missing = [k for k in SETTING_KEYS if k not in values]
    if missing:
        raise SetupError(f"{path}: missing {', '.join(missing)}")
    if values["correction"] not in CORRECTION_WORDS:
        raise SetupError(f"{path}: correction = {values['correction']!r}; "
                         f"choose one of {', '.join(CORRECTION_WORDS)}")
    for key in ("ls_weights", "average_weights"):
        if values[key] not in WEIGHT_CHOICES:
            raise SetupError(f"{path}: {key} = {values[key]!r}; "
                             f"choose one of {', '.join(WEIGHT_CHOICES)}")
    return Settings(values["correction"], values["ls_weights"],
                    values["average_weights"],
                    parse_nprocs(values["np"], f"{path}: np"))


def parse_nprocs(text, what="np"):
    """np as a positive integer written with the ASCII digits 0-9 only
    (int() would also accept '4_0', '+4' or other Unicode digits)."""
    text = str(text).strip()
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise SetupError(f"{what} = {text!r}; give a positive integer "
                         "written with the digits 0-9, for example 4 "
                         "(1 = serial run)")
    return int(text)


def is_paper_setting(settings):
    """True for the scheme of the convergence tables of the paper (np does not matter)."""
    return (settings.correction == PAPER_SETTING["correction"]
            and settings.ls_weights == PAPER_SETTING["ls_weights"]
            and settings.average_weights == PAPER_SETTING["average_weights"])


def describe_settings(settings):
    return (f"correction = {settings.correction}, "
            f"ls_weights = {settings.ls_weights}, "
            f"average_weights = {settings.average_weights}, "
            f"np = {settings.nprocs}")


# -------------------------------------------------------------- environment
def check_environment(nprocs, solver):
    """Stop with a clear message if OpenFOAM, MPI or the solver is missing."""
    problems = []
    if not os.environ.get("WM_PROJECT_DIR"):
        problems.append(
            "the OpenFOAM environment is not loaded (WM_PROJECT_DIR is not "
            "set); load the MPI and OpenFOAM environment as in step (0) of "
            "README.txt")
    else:
        tools = ("decomposePar", "mpirun") if nprocs > 1 else ()
        for tool in tools:
            if shutil.which(tool) is None:
                problems.append(
                    f"'{tool}' is not on PATH (needed for np = {nprocs}); "
                    "load the MPI and OpenFOAM environment as in step (0) "
                    "of README.txt, or use --np 1 for a serial run")
    if not (os.path.isfile(solver) and os.access(solver, os.X_OK)):
        problems.append(f"the solver {solver} does not exist; build it with "
                        "'solver/Allwmake' in the repository directory")
    if problems:
        raise SetupError("\n  ".join(["setup incomplete:"] + problems))
    # The solver must start (shared libraries of OpenFOAM and MPI found).
    try:
        proc = subprocess.run([solver, "-help"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise SetupError(f"the solver {solver} does not start: {err}")
    if proc.returncode != 0:
        tail = proc.stdout.decode(errors="replace").strip().splitlines()[-15:]
        raise SetupError(f"the solver {solver} does not start (exit code "
                         f"{proc.returncode}):\n    " + "\n    ".join(tail))


def openfoam_version():
    return os.environ.get("WM_PROJECT_VERSION", "unknown")


# ----------------------------------------------------- reading the polyMesh
PolyMesh = collections.namedtuple(
    "PolyMesh", "points face_sizes face_labels owner neighbour n_cells "
                "has_empty_patch")


def _read_text(path):
    if os.path.isfile(path):
        with open(path, encoding="latin-1") as f:
            return f.read()
    if os.path.isfile(path + ".gz"):
        with gzip.open(path + ".gz", "rt", encoding="latin-1") as f:
            return f.read()
    raise SetupError(f"mesh file not found: {path}")


def _foam_file(path):
    """(header entries, text after the header) of an ASCII OpenFOAM file."""
    text = _read_text(path)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    m = re.search(r"\bFoamFile\s*\{([^}]*)\}", text)
    if m is None:
        raise SetupError(f"{path}: no FoamFile header")
    header = {k: v.strip().strip('"')
              for k, v in re.findall(r"(\w+)\s+([^;]*);", m.group(1))}
    if header.get("format", "ascii") != "ascii":
        raise SetupError(f"{path}: format {header['format']}; the runners "
                         "read ASCII meshes only (writeFormat ascii)")
    return header, text[m.end():]


def _numbers(text, dtype, count, path):
    """count numbers separated by white space."""
    if count == 0:
        return np.zeros(0, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            values = np.fromstring(text, dtype=dtype, sep=" ")   # fast path
        except (ValueError, DeprecationWarning):
            raise SetupError(f"{path}: unexpected characters in a list")
        except (AttributeError, TypeError):    # numpy without fromstring
            try:
                values = np.array(text.split(), dtype=dtype)
            except ValueError:
                raise SetupError(f"{path}: unexpected characters in a list")
    if values.size != count:
        raise SetupError(f"{path}: {values.size} numbers, expected {count}")
    return values


def _list_start(text, path):
    """Size n and position after '(' (or after '{' for n{value}) of a list."""
    m = re.match(r"\s*(\d+)\s*([({])", text)
    if m is None:
        raise SetupError(f"{path}: no list found")
    return int(m.group(1)), m.group(2), m.end()


def _label_list(text, path):
    """Flat list of integers, written as n(...) or as n{value}; returns the
    labels and the text after the list."""
    n, bracket, pos = _list_start(text, path)
    if bracket == "{":
        end = text.index("}", pos)
        value = _numbers(text[pos:end], np.int64, 1, path)[0]
        return np.full(n, value, dtype=np.int64), text[end + 1:]
    end = text.index(")", pos)
    return _numbers(text[pos:end], np.int64, n, path), text[end + 1:]


def _faces(path):
    header, text = _foam_file(path)
    if header.get("class") == "faceCompactList":
        offsets, rest = _label_list(text, path)
        labels, _ = _label_list(rest, path)
        sizes = np.diff(offsets)
    else:
        n, bracket, pos = _list_start(text, path)
        if bracket != "(":
            raise SetupError(f"{path}: unexpected face list")
        inner = text[pos:text.rindex(")")]
        sizes = np.array(re.findall(r"(\d+)\s*\(", inner), dtype=np.int64)
        if sizes.size != n:
            raise SetupError(f"{path}: {sizes.size} faces, expected {n}")
        flat = re.sub(r"\d+\s*\(", " ", inner).replace(")", " ")
        labels = _numbers(flat, np.int64, int(sizes.sum()), path)
    if sizes.size and sizes.min() < 3:
        raise SetupError(f"{path}: a face with fewer than 3 vertices")
    return sizes, labels


def _points(path):
    _, text = _foam_file(path)
    n, bracket, pos = _list_start(text, path)
    if bracket != "(":
        raise SetupError(f"{path}: unexpected point list")
    inner = text[pos:text.rindex(")")]
    flat = inner.replace("(", " ").replace(")", " ")
    return _numbers(flat, np.float64, 3 * n, path).reshape(n, 3)


def read_polymesh(polymesh_dir):
    """Points, faces, owner and neighbour of an ASCII OpenFOAM polyMesh."""
    points = _points(os.path.join(polymesh_dir, "points"))
    sizes, labels = _faces(os.path.join(polymesh_dir, "faces"))
    owner, _ = _label_list(_foam_file(os.path.join(polymesh_dir, "owner"))[1],
                           os.path.join(polymesh_dir, "owner"))
    neighbour, _ = _label_list(
        _foam_file(os.path.join(polymesh_dir, "neighbour"))[1],
        os.path.join(polymesh_dir, "neighbour"))
    _, boundary = _foam_file(os.path.join(polymesh_dir, "boundary"))
    if owner.size != sizes.size or neighbour.size > sizes.size:
        raise SetupError(f"{polymesh_dir}: owner/neighbour do not match faces")
    if labels.size and (labels.min() < 0 or labels.max() >= len(points)):
        raise SetupError(f"{polymesh_dir}: face vertex out of range")
    n_cells = int(max(owner.max(), neighbour.max(initial=-1))) + 1
    has_empty = re.search(r"\btype\s+empty\s*;", boundary) is not None
    return PolyMesh(points, sizes, labels, owner, neighbour, n_cells,
                    has_empty)


# ----------------------------------------------------------- mesh size h
_CHUNK_BYTES = 32 * 2 ** 20   # bound on the work arrays of one chunk


def cell_diameters(mesh, dim):
    """h_K = largest distance between two vertices of the cell K, for every
    cell; only the first dim coordinates are used (dim = 2: x and y)."""
    n_points = len(mesh.points)
    n_faces = mesh.face_sizes.size
    n_internal = mesh.neighbour.size
    # (cell, vertex) for every vertex of every face, on both sides of the face
    face_of = np.repeat(np.arange(n_faces), mesh.face_sizes)
    internal = face_of < n_internal
    cell = np.concatenate([mesh.owner[face_of],
                           mesh.neighbour[face_of[internal]]]).astype(np.int64)
    vertex = np.concatenate([mesh.face_labels,
                             mesh.face_labels[internal]]).astype(np.int64)
    # vertex set of each cell: unique pairs, sorted by cell
    key = np.unique(cell * n_points + vertex)
    cell = key // n_points
    vertex = key - cell * n_points
    count = np.bincount(cell, minlength=mesh.n_cells)
    if count.min() < 2:
        raise SetupError("a cell with fewer than two vertices")
    first = np.zeros(mesh.n_cells, dtype=np.int64)
    first[1:] = np.cumsum(count)[:-1]
    x = mesh.points[:, :dim]
    h = np.empty(mesh.n_cells)
    # cells with the same number n of vertices together: n x n distances
    for n in np.unique(count):
        cells = np.flatnonzero(count == n)
        step = max(1, _CHUNK_BYTES // (int(n) * int(n) * dim * 8))
        for s in range(0, cells.size, step):
            c = cells[s:s + step]
            xc = x[vertex[first[c][:, None] + np.arange(n)]]    # (m, n, dim)
            d2 = ((xc[:, :, None, :] - xc[:, None, :, :]) ** 2).sum(-1)
            h[c] = np.sqrt(d2.reshape(c.size, -1).max(axis=1))
    return h


def mesh_size(level_dir, test):
    """(number of cells, number of facets, h = max_K h_K) of the mesh of one
    level.  The facets are those of the mesh in the sense of the paper: in
    2D the front and back faces of the slab (one of each per cell) are not
    facets."""
    polymesh = os.path.join(level_dir, "constant", "polyMesh")
    mesh = read_polymesh(polymesh)
    n_facets = mesh.face_sizes.size
    if test.dim == 2:
        if not mesh.has_empty_patch or np.unique(mesh.points[:, 2]).size != 2:
            raise SetupError(f"{polymesh}: not a 2D mesh (one cell thick in "
                             "z, with an empty patch); is this a "
                             f"{test.name} mesh?")
        n_facets -= 2 * mesh.n_cells
    elif mesh.has_empty_patch:
        raise SetupError(f"{polymesh}: has an empty patch (a 2D mesh); is "
                         f"this a {test.name} mesh?")
    return (mesh.n_cells, n_facets,
            float(cell_diameters(mesh, test.dim).max()))


# ------------------------------------------------------------ case set-up
def level_dir(meshes, test, level):
    return os.path.join(meshes, test.key, f"N{level}")


def case_dir(results, test, level, nprocs):
    return os.path.join(results, test.key, f"N{level}", f"np{nprocs}")


def check_meshes(meshes, test):
    """Every level must be a complete mesh of the mesh script: the polyMesh
    files and MESH.md, which the script writes after the last step.  The
    message names the commands that make the missing levels only."""
    missing, incomplete = [], []
    for level in LEVELS:
        ldir = level_dir(meshes, test, level)
        polymesh = os.path.join(ldir, "constant", "polyMesh")
        if not os.path.lexists(ldir):
            missing.append(level)
        elif not (all(os.path.isfile(os.path.join(polymesh, f))
                      or os.path.isfile(os.path.join(polymesh, f + ".gz"))
                      for f in POLYMESH_FILES)
                  and os.path.isfile(os.path.join(ldir, MESH_DONE))):
            incomplete.append(level)
    if not (missing or incomplete):
        return
    lines = []
    if missing:
        lines.append("mesh not found:" + "".join(
            "\n    " + level_dir(meshes, test, k) for k in missing))
    if incomplete:
        lines.append(
            f"incomplete mesh (a constant/polyMesh file or {MESH_DONE} is "
            f"missing: {test.mesh_script} stopped before the end, or did "
            "not make the directory):" + "".join(
                "\n    " + level_dir(meshes, test, k) for k in incomplete))
    make = f"python {test.mesh_script} --levels " + " ".join(
        str(k) for k in sorted(missing + incomplete))
    if os.path.realpath(meshes) != os.path.realpath(DEFAULT_MESHES):
        make += " --out " + shlex.quote(meshes)
    commands = []
    if incomplete:
        commands.append("rm -rf " + " ".join(
            shlex.quote(level_dir(meshes, test, k)) for k in incomplete))
    commands.append(make)
    raise SetupError("\n  ".join(lines) + f"\n  make these {test.name} "
                     "meshes first, in the repository directory:\n    "
                     + "\n    ".join(commands))


def mesh_report(test, n_cells, n_facets):
    """(meshes equal the paper's, report lines): the numbers of cells and
    facets against |P_h| and |F_h| in the mesh table of the paper."""
    same_all, lines = True, []
    for k in range(len(n_cells)):
        want = (test.paper_cells[k], test.paper_facets[k])
        got = f"{n_cells[k]} cells, {n_facets[k]} facets"
        if (n_cells[k], n_facets[k]) == want:
            lines.append(f"  N{k + 1}  same       {got}")
        else:
            same_all = False
            lines.append(f"  N{k + 1}  DIFFERENT  {got} (paper: {want[0]} "
                         f"cells, {want[1]} facets)")
    if not same_all:
        lines.append(
            f"The meshes are not those of the paper, so the table is not "
            f"expected to equal {test.table} in the paper.  Make the meshes with "
            f"{test.mesh_script} and gmsh 4.15.2 (requirements.txt); other "
            "gmsh versions give other meshes.")
    return same_all, lines


def _dictionary(path, object_name, comment, entries):
    lines = [
        "/*" + "-" * 75 + "*\\",
        *("    " + c for c in comment),
        "\\*" + "-" * 75 + "*/",
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        f"    object      {object_name};",
        "}",
        "",
        *(f"{k:<20}{v};" for k, v in entries),
        "",
    ]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def _remove_case(case):
    """Remove a case written earlier by this module (never follows links)."""
    mark = os.path.join(case, CASE_MARK)
    ours = False
    if os.path.isdir(case) and not os.path.islink(case) and os.path.isfile(mark):
        with open(mark, encoding="utf-8", errors="replace") as f:
            ours = f.readline().strip() == CASE_MARK_FIRST_LINE
    if not ours:
        raise SetupError(f"{case} exists but was not written by the runners; "
                         "remove it or choose another --out")
    link = os.path.join(case, "constant", "polyMesh")
    if os.path.islink(link):
        os.unlink(link)
    shutil.rmtree(case)


def prepare_case(case, level, test, settings, solver):
    """Write the case directory of one run; returns the commands to run."""
    if os.path.lexists(case):
        _remove_case(case)
    os.makedirs(os.path.join(case, "constant"))
    polymesh = os.path.realpath(os.path.join(level, "constant", "polyMesh"))
    commands = []
    if settings.nprocs > 1:
        commands.append(["decomposePar", "-force", "-case", case])
        commands.append(["mpirun", "-np", str(settings.nprocs), solver,
                         "-parallel", "-case", case])
    else:
        commands.append([solver, "-case", case])
    with open(os.path.join(case, CASE_MARK), "w", encoding="utf-8") as f:
        f.write(CASE_MARK_FIRST_LINE + "\n\n"
                f"test      : {test.name} ({test.problem})\n"
                f"mesh      : {level}\n"
                f"settings  : {describe_settings(settings)}\n"
                f"written   : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
                "commands  :\n"
                + "".join("    " + " ".join(c) + "\n" for c in commands))
    os.symlink(polymesh, os.path.join(case, "constant", "polyMesh"))
    shutil.copytree(CASE_SYSTEM, os.path.join(case, "system"))
    _dictionary(
        os.path.join(case, "system", "ccfvDict"), "ccfvDict",
        ["Solver dictionary of this case, written by the convergence runner",
         "from settings.txt (see the comments in settings.txt for the",
         "formulas of correction, lsWeights and averageWeights)."],
        [("solution", test.key),
         ("correction", CORRECTION_WORDS[settings.correction]),
         ("lsWeights", settings.ls_weights),
         ("averageWeights", settings.average_weights),
         ("epsilon", EPSILON),
         ("maxOuter", str(MAX_OUTER)),
         ("writeFields", "false")])
    if settings.nprocs > 1:
        _dictionary(
            os.path.join(case, "system", "decomposeParDict"),
            "decomposeParDict",
            ["Decomposition of the mesh for the parallel run, written by the",
             "convergence runner."],
            [("numberOfSubdomains", str(settings.nprocs)),
             ("method", "scotch")])
    return commands


# --------------------------------------------------------------- running
def _tail(path, n=20):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().rstrip().splitlines()
    except OSError:
        return ""
    return "\n    ".join([""] + lines[-n:])


def _run(command, log):
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(command, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def run_case(case, commands, nprocs):
    """Run decomposePar (np > 1) and the solver; returns the wall time."""
    start = time.time()
    if nprocs > 1:
        log = os.path.join(case, "log.decomposePar")
        rc = _run(commands[0], log)
        parts = [os.path.join(case, f"processor{k}", "constant", "polyMesh",
                              "owner") for k in range(nprocs)]
        if rc != 0 or not all(os.path.isfile(p) for p in parts):
            raise RunError(f"decomposePar failed (exit code {rc}), see {log}"
                           + _tail(log))
    log = os.path.join(case, "log.run")
    rc = _run(commands[-1], log)
    if rc != 0:
        hint = ""
        if nprocs > 1:
            hint = (f"\n  If MPI cannot start {nprocs} processes on this "
                    "machine, run serially with --np 1 or choose a smaller "
                    "np (--np or settings.txt).")
        raise RunError(f"the solver failed (exit code {rc}), see {log}"
                       + _tail(log) + hint)
    return time.time() - start


RESULT_INT = ("dim", "nProcs", "nCells", "K", "linIter", "diverged")
RESULT_FLOAT = ("res", "E2", "H1h", "Linf", "time")
RESULT_WORD = ("solution", "correction", "lsWeights", "averageWeights")


def parse_result(log):
    """key=value pairs of the single line of log that starts with 'RESULT '."""
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f if line.startswith("RESULT ")]
    if len(lines) != 1:
        raise RunError(f"{log}: {len(lines)} lines start with 'RESULT ', "
                       "expected exactly one")
    pairs = dict(re.findall(r"(\w+)=(\S+)", lines[0]))
    missing = [k for k in RESULT_INT + RESULT_FLOAT + RESULT_WORD
               if k not in pairs]
    if missing:
        raise RunError(f"{log}: the RESULT line has no {', '.join(missing)}")
    result = {"line": lines[0]}
    try:
        for k in RESULT_INT:
            result[k] = int(pairs[k])
        for k in RESULT_FLOAT:
            result[k] = float(pairs[k])
    except ValueError as err:
        raise RunError(f"{log}: unreadable RESULT value ({err})")
    for k in RESULT_WORD:
        result[k] = pairs[k]
    return result


def check_result(result, test, settings, n_cells, log):
    """Check that the solver ran what was asked; returns True if converged."""
    want = {"dim": test.dim, "nProcs": settings.nprocs, "nCells": n_cells,
            "solution": test.key,
            "correction": CORRECTION_WORDS[settings.correction],
            "lsWeights": settings.ls_weights,
            "averageWeights": settings.average_weights}
    wrong = [f"{k}={result[k]} (expected {v})" for k, v in want.items()
             if result[k] != v]
    if wrong:
        raise RunError(f"{log}: the RESULT line does not match the case: "
                       + ", ".join(wrong))
    return (result["K"] > 0 and result["diverged"] == 0
            and result["res"] < float(EPSILON))


# ---------------------------------------------------------- the table
def convergence_orders(h, errors):
    """log(E_N/E_{N-1}) / log(h_N/h_{N-1}); None for the first mesh."""
    orders = [None]
    for k in range(1, len(h)):
        try:
            orders.append(math.log(errors[k] / errors[k - 1])
                          / math.log(h[k] / h[k - 1]))
        except (ValueError, ZeroDivisionError):
            orders.append(float("nan"))
    return orders


Table = collections.namedtuple("Table", "h e_h o_h e_0 o_0")


def make_table(h, e_h, e_0):
    """h: mesh sizes, e_h: |phi_I^* - phi_h|_h, e_0: ||phi_I^* - phi_h||_0."""
    return Table(list(h), list(e_h), convergence_orders(h, e_h),
                 list(e_0), convergence_orders(h, e_0))


def _order(value):
    return "---" if value is None else f"{value:.3f}"


def latex_rows(table):
    """Rows of the table body as in the LaTeX source of the paper."""
    return [f"\t\t\t\t{table.h[k]:.3f} & {table.e_h[k]:.3e} & "
            f"{_order(table.o_h[k])} & {table.e_0[k]:.3e} & "
            f"{_order(table.o_0[k])} \\\\" for k in range(len(table.h))]


def text_table(test, table):
    """The table laid out like the paper, in plain text."""
    groups = ("|phi_I^* - phi_h|_h", "||phi_I^* - phi_h||_{0,Omega}")
    w_h, w_e, w_o, gap = 7, 16, 13, 2      # columns h, error, order
    span = w_e + w_o                       # one group: error and order

    def line(first, groups_text):
        return (first.center(w_h)
                + "".join(" " * gap + g for g in groups_text)).rstrip()

    rule = "-" * (w_h + 2 * (gap + span))
    lines = [test.title, rule,
             line("", [g.center(span) for g in groups]),
             line("h", ["Error".center(w_e) + "Order".center(w_o)] * 2),
             rule]
    for k in range(len(table.h)):
        lines.append(line(f"{table.h[k]:.3f}", [
            f"{table.e_h[k]:.3e}".center(w_e) + _order(table.o_h[k]).center(w_o),
            f"{table.e_0[k]:.3e}".center(w_e) + _order(table.o_0[k]).center(w_o)]))
    lines.append(rule)
    return lines


def full_precision_lines(table, n_cells):
    def num(x):
        return "---" if x is None else f"{x:.10e}"
    head = ("N", "cells", "h", "|.|_h error", "order",
            "||.||_0 error", "order")
    rows = [head] + [(f"N{k + 1}", str(n_cells[k]), num(table.h[k]),
                      num(table.e_h[k]), num(table.o_h[k]),
                      num(table.e_0[k]), num(table.o_0[k]))
                     for k in range(len(table.h))]
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    return ["  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
            for r in rows]


def read_expected(path):
    """Table rows of an expected/ file (lines starting with '#' skipped)."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f
                if line.strip() and not line.lstrip().startswith("#")]


def compare_rows(rows, expected):
    """(same, report lines); a row is the same when it equals the paper row
    up to leading and trailing white space."""
    report, same = [], len(rows) == len(expected)
    for k in range(max(len(rows), len(expected))):
        got = rows[k].strip() if k < len(rows) else "(missing)"
        want = expected[k].strip() if k < len(expected) else "(missing)"
        if got == want:
            report.append(f"  N{k + 1}  same       {got}")
        else:
            same = False
            report.append(f"  N{k + 1}  DIFFERENT  computed: {got}")
            report.append(f"                 paper   : {want}")
    return same, report


# ------------------------------------------------------------ the runner
def run_test(test, meshes=DEFAULT_MESHES, results=DEFAULT_RESULTS,
             settings_file=DEFAULT_SETTINGS, nprocs=None, solver=SOLVER):
    """Run one convergence test and print its table; returns the exit code."""
    meshes = os.path.abspath(meshes)
    results = os.path.abspath(results)
    solver = os.path.abspath(solver)
    settings = read_settings(os.path.abspath(settings_file))
    if nprocs is not None:
        settings = settings._replace(nprocs=parse_nprocs(str(nprocs), "--np"))
    if os.path.realpath(results) == os.path.realpath(meshes):
        raise SetupError("--out must differ from --meshes")
    check_environment(settings.nprocs, solver)
    check_meshes(meshes, test)
    paper = is_paper_setting(settings)

    print(f"{test.name}: {test.problem}")
    print(f"settings : {describe_settings(settings)}"
          + ("  (setting of the paper)" if paper else ""))
    print(f"meshes   : {os.path.join(meshes, test.key)}")
    print(f"results  : {os.path.join(results, test.key)}")
    print(f"OpenFOAM : {openfoam_version()}, solver {solver}"
          + (f", mpirun -np {settings.nprocs}" if settings.nprocs > 1
             else ", serial"))
    if openfoam_version().lstrip("v") != "2606":
        print("note     : the paper used OpenFOAM v2606; other versions are "
              "not tested")
    if settings.nprocs > (os.cpu_count() or 1):
        print(f"note     : np = {settings.nprocs} is more than the "
              f"{os.cpu_count()} CPUs of this machine")
    print()

    h, n_cells, n_facets, runs = [], [], [], []
    for level in LEVELS:
        ldir = level_dir(meshes, test, level)
        cells, facets, h_level = mesh_size(ldir, test)
        paper_mesh = (test.paper_cells[level - 1],
                      test.paper_facets[level - 1])
        if (cells, facets) != paper_mesh:
            print(f"N{level}: note: {cells} cells and {facets} facets; the "
                  f"mesh of the paper has {paper_mesh[0]} cells and "
                  f"{paper_mesh[1]} facets ({test.mesh_table} in the paper)")
        case = case_dir(results, test, level, settings.nprocs)
        print(f"N{level}: {cells} cells, h = {h_level:.6f}; running ...",
              end="", flush=True)
        commands = prepare_case(case, ldir, test, settings, solver)
        wall = run_case(case, commands, settings.nprocs)
        log = os.path.join(case, "log.run")
        result = parse_result(log)
        converged = check_result(result, test, settings, cells, log)
        with open(os.path.join(case, CASE_MARK), "a", encoding="utf-8") as f:
            f.write(f"status    : {'converged' if converged else 'NOT CONVERGED'}"
                    f"\n{result['line']}\n")
        print(f" K = {result['K']}, res = {result['res']:.2e}, "
              f"{wall:.1f} s" + ("" if converged else "  NOT CONVERGED"))
        h.append(h_level)
        n_cells.append(cells)
        n_facets.append(facets)
        runs.append((result, converged))

    table = make_table(h, [r["H1h"] for r, _ in runs],
                       [r["E2"] for r, _ in runs])
    rows = latex_rows(table)
    out = [f"{test.name} ({test.problem})",
           f"settings: {describe_settings(settings)}"
           + ("  (setting of the paper)" if paper else ""),
           f"OpenFOAM {openfoam_version()}, written "
           f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    out += text_table(test, table)
    out += ["", "LaTeX rows (as in the source of the paper):"] + rows
    out += ["", "Full-precision values (the orders are computed from "
                "these h and errors):"]
    out += full_precision_lines(table, n_cells)
    out += ["", "Outer iterations of the solver (K), final residual (res):"]
    for k, (r, conv) in enumerate(runs):
        out.append(f"  N{k + 1}  K = {r['K']:<4d} res = {r['res']:.3e}  "
                   f"linear iterations = {r['linIter']:<6d}"
                   f"solver time = {r['time']:.3f} s"
                   + ("" if conv else "  NOT CONVERGED"))
    out.append("")
    meshes_same, report = mesh_report(test, n_cells, n_facets)
    out.append(f"Meshes compared with {test.mesh_table} in the paper "
               "(|P_h| cells, |F_h| facets):")
    out += report
    out.append("")

    all_converged = all(conv for _, conv in runs)
    if not all_converged:
        out.append("Some runs did not converge (see above); their errors "
                   "are not reliable.")
    if paper:
        expected_file = os.path.join(EXPECTED_DIR, test.expected)
        same, report = compare_rows(rows, read_expected(expected_file))
        out.append(f"Comparison with {test.table} in the paper "
                   f"(convergence/expected/{test.expected}):")
        out += report
        passed = same and all_converged
        if passed:
            out.append(f"PASS: the table is identical to {test.table} in "
                       "the paper.")
        elif not same:
            out.append(f"FAIL: the table differs from {test.table} in the "
                       "paper." + ("" if meshes_same else "  The meshes "
                                   "differ from the meshes of the paper "
                                   "(see above)."))
        else:
            out.append(f"FAIL: the rows equal {test.table} in the paper, "
                       "but not every run converged.")
        status = 0 if passed else 1
    else:
        out.append(f"No comparison with {test.table} is made: the paper uses "
                   "correction = over-relaxed, ls_weights = iii, "
                   "average_weights = iii.")
        status = 0 if all_converged else 1

    text = "\n".join(out) + "\n"
    table_file = os.path.join(results, test.key, "table.txt")
    with open(table_file, "w", encoding="utf-8") as f:
        f.write(text)
    print()
    print(text, end="")
    print(f"(written to {table_file})")
    return status


def main(test, argv=None, solver=SOLVER):
    """Command-line interface shared by run_example1_2d.py and run_example2_3d.py."""
    parser = argparse.ArgumentParser(
        description=f"Run {test.name} on the meshes N = 1, ..., 4 "
                    f"and print its convergence table ({test.problem}).",
        epilog="Exit status: 0 = all runs converged and, for the setting of "
               "the paper, the table equals the paper; 1 = a run failed or "
               "did not converge, or the table differs; 2 = setup incomplete.")
    parser.add_argument("--meshes", metavar="DIR", default=DEFAULT_MESHES,
                        help="directory with the mesh levels "
                             f"{test.key}/N1 ... {test.key}/N4 "
                             "(default: meshes/ in the repository)")
    parser.add_argument("--out", metavar="DIR", default=DEFAULT_RESULTS,
                        help="directory for the cases and the table "
                             f"{test.key}/table.txt "
                             "(default: results/ in the repository)")
    parser.add_argument("--settings", metavar="FILE", default=DEFAULT_SETTINGS,
                        help="settings file (default: settings.txt in the "
                             "repository)")
    parser.add_argument("--np", metavar="N", default=None,
                        help="number of MPI processes, overrides np of the "
                             "settings file (1 = serial run)")
    args = parser.parse_args(argv)
    try:
        return run_test(test, meshes=args.meshes, results=args.out,
                        settings_file=args.settings, nprocs=args.np,
                        solver=solver)
    except SetupError as err:
        print(f"\nerror: {err}", file=sys.stderr)
        return 2
    except RunError as err:
        print(f"\nerror: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
