#!/usr/bin/env python3
# Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
"""Generate the polyhedral meshes of Example 2 (the 3D convergence test of the paper).

usage: python mesh/make_mesh_3d.py [--out DIR] [--levels 1 2 3 4] [--figures]

Level N is written to DIR/example2/N<N> (default DIR: meshes/ in the repository)
as an OpenFOAM case directory: constant/polyMesh, system/, MESH.md and logs/,
plus figures/ with --figures.  An existing level directory is never touched;
remove it to make that level again.  A level is built in DIR/example2/N<N>.partial
and renamed to N<N> only after its last step, so N<N> always holds a complete
mesh.  If a step fails or the script is interrupted, N<N>.partial stays for
inspection (logs/ has the log of every step) and the next run replaces it.

Construction of level N, with the target edge length lc = 0.19 / N:
  1. gmsh 4.15.2 with one thread: tetrahedral mesh of the box (-1.25, 1.25)^3
     with target edge length lc; Frontal-Delaunay triangles on the sides
     (Mesh.Algorithm 6), Delaunay tetrahedra inside (Mesh.Algorithm3D 1),
     Mesh.Optimize 1 and Mesh.OptimizeNetgen 1.  Written as an MSH 2.2 ascii
     file with the physical volume 'internal' and the physical surface
     'boundary' (all six sides);
  2. OpenFOAM gmshToFoam: the tetrahedral mesh in OpenFOAM format;
  3. OpenFOAM polyDualMesh 60 -overwrite: the dual polyhedral mesh, one cell
     per vertex of the tetrahedral mesh; the edges and corners of the box
     (feature angle 60 degrees) are kept;
  4. OpenFOAM combinePatchFaces 1 -overwrite: the boundary faces of a cell on
     one side of the box become a single face;
  5. OpenFOAM renumberMesh -overwrite (Cuthill-McKee order of the cells);
  6. check_mesh.py: corner, edge and side rules; a failing mesh is moved to
     constant/polyMesh.rejected and the script stops with an error.

The target edge lengths are passed to gmsh as the decimal strings used for the
paper, "0.19", "0.095", "0.0633333333333333" and "0.0475", that is 0.19/N with
15 significant digits.  Do not compute 0.19/3 instead: in double precision it
is 3 units in the last place larger than "0.0633333333333333" and gmsh then
makes a different mesh.

Run the script in a shell with the OpenFOAM v2606 environment (gmshToFoam,
polyDualMesh, combinePatchFaces and renumberMesh on PATH) and with a Python
that has the packages of requirements.txt.  Every step runs in its own
process; its log, wall time and peak memory are kept in logs/ and in MESH.md.
"""
import argparse
import datetime
import glob
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HALF = 1.25                         # the box is (-HALF, HALF)^3
# target edge length 0.19 / N, as the decimal strings used for the paper
TARGET = {1: "0.19", 2: "0.095", 3: "0.0633333333333333", 4: "0.0475"}
# polyDualMesh: boundary edges where the normal turns by more than 60 degrees
# (the box edges) are kept as features
FEATURE_ANGLE = "60"
# combinePatchFaces: boundary faces of one cell that meet at less than 1 degree are merged
MERGE_ANGLE = "1"
GMSH_VERSION = "4.15.2"
OPENFOAM_TOOLS = ["gmshToFoam", "polyDualMesh", "combinePatchFaces", "renumberMesh"]
# environment of the steps: serial numerics, no .pyc files written next to the scripts
SERIAL = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          "PYTHONDONTWRITEBYTECODE": "1"}
# a level is built in N<N>.partial, which holds this marker file until it is complete
PARTIAL_SUFFIX = ".partial"
PARTIAL_MARK = "INCOMPLETE.txt"
PARTIAL_TEXT = "Level under construction by mesh/make_mesh_3d.py; removed by its next run.\n"


# ---------------------------------------------------------------------------
# Step 1 (own process): tetrahedral mesh of the box with gmsh
# ---------------------------------------------------------------------------
def tetrahedra(lc_text, msh_path):
    """Tetrahedral mesh of the box with target edge length lc, as MSH 2.2 ascii."""
    import gmsh

    lc = float(lc_text)
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("box")
    gmsh.model.occ.addBox(-HALF, -HALF, -HALF, 2 * HALF, 2 * HALF, 2 * HALF)
    gmsh.model.occ.synchronize()

    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(3, volumes, name="internal")
    gmsh.model.addPhysicalGroup(2, surfaces, name="boundary")    # one patch for the six sides

    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)                # uniform target size lc
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)                   # sides: Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)                 # volume: Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("General.NumThreads", 1)               # the mesh depends on the thread count
    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(msh_path)

    stats = {"gmsh": gmsh.__version__, "lc": lc_text, "vertices": len(gmsh.model.mesh.getNodes()[0]),
             "tetrahedra": sum(len(t) for t in gmsh.model.mesh.getElements(3)[1]),
             "boundary_triangles": sum(len(t) for t in gmsh.model.mesh.getElements(2)[1])}
    gmsh.finalize()
    print("STATS " + json.dumps(stats))
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class StepFailed(Exception):
    pass


def neutral(cmd, case):
    """The command line with the case directory and the interpreter written generically."""
    out = []
    for word in cmd:
        word = str(word)
        if word == sys.executable:
            word = "python"
        elif word.startswith(HERE + os.sep):
            word = "mesh/" + os.path.relpath(word, HERE)
        elif word.startswith(case):
            word = "<case>" + word[len(case):]
        out.append(shlex.quote(word) if re.search(r"\s", word) else word)
    return " ".join(out)


def log_tail(log, n=12):
    """The last n non-empty lines of a log, indented, for an error message."""
    try:
        with open(log, errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    except OSError:
        return ""
    return "".join("\n    | " + ln for ln in lines[-n:])


def run_step(name, cmd, case, log, steps, cwd=None):
    """Run one step in its own process; record wall time and peak memory (ru_maxrss)."""
    start = time.perf_counter()
    with open(log, "w") as out:
        child = subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=subprocess.STDOUT, cwd=cwd,
                                 env=dict(os.environ, **SERIAL))
        try:
            _, status, usage = os.wait4(child.pid, 0)
        except KeyboardInterrupt:                       # Ctrl-C: stop the step as well
            child.kill()
            child.wait()
            raise
    child.returncode = os.waitstatus_to_exitcode(status)
    peak_kib = usage.ru_maxrss / 1024 if sys.platform == "darwin" else usage.ru_maxrss
    steps.append({"step": name, "command": neutral(cmd, case), "log": os.path.relpath(log, case),
                  "wall_s": round(time.perf_counter() - start, 3),
                  "cpu_s": round(usage.ru_utime + usage.ru_stime, 3),
                  "peak_rss_kib": int(peak_kib), "peak_rss_mib": round(peak_kib / 1024, 1),
                  "exit_status": child.returncode})
    if child.returncode != 0:
        raise StepFailed(f"step '{name}' failed with exit status {child.returncode}; "
                         f"the last lines of {log}:" + log_tail(log))


def stats_of(log):
    """The STATS line printed by an internal step."""
    with open(log) as fh:
        lines = [ln for ln in fh if ln.startswith("STATS ")]
    return json.loads(lines[-1][6:]) if lines else {}


def last_line(log):
    with open(log) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    return lines[-1] if lines else ""


def mesh_sizes(case):
    """Counts from the owner header and the patches of the final polyMesh."""
    pm = os.path.join(case, "constant", "polyMesh")
    with open(os.path.join(pm, "owner")) as fh:
        head = fh.read(4096)
    note = dict(re.findall(r"(\w+):(\d+)", re.search(r'note\s+"([^"]*)"', head).group(1)))
    with open(os.path.join(pm, "boundary")) as fh:
        text = fh.read()
    patches = [(m.group(1), m.group(2), int(m.group(3))) for m in
               re.finditer(r"(\w+)\s*\{\s*type\s+(\w+);[^}]*?nFaces\s+(\d+);", text)]
    return {k: int(v) for k, v in note.items()}, patches


def openfoam_build(log):
    with open(log) as fh:
        m = re.search(r"^Build\s*:\s*(.*)$", fh.read(), re.M)
    return m.group(1).strip() if m else "unknown"


def gmsh_version():
    """Import gmsh in a separate process started like the gmsh step; returns its version
    ('' if the module has no version attribute) or stops with the import error."""
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--gmsh-version"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          env=dict(os.environ, **SERIAL))
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        raise SystemExit(f"make_mesh_3d: 'import gmsh' fails for {sys.executable}:\n  "
                         + (lines[-1] if lines else f"exit status {proc.returncode}")
                         + "\n  Install the Python packages (python -m pip install -r requirements.txt) and "
                         "the system libraries that the gmsh wheel needs (README.txt, Installation).")
    return proc.stdout.strip()


def check_prerequisites(figures):
    missing = [t for t in OPENFOAM_TOOLS if shutil.which(t) is None]
    if missing:
        raise SystemExit("make_mesh_3d: OpenFOAM utilities not found on PATH: " + ", ".join(missing) +
                         "\n  Load the OpenFOAM v2606 environment first, as in step (0) of README.txt.")
    for module in ["numpy"] + (["matplotlib"] if figures else []):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"make_mesh_3d: the Python package '{module}' is missing for {sys.executable}"
                             "\n  Install the packages: python -m pip install -r requirements.txt")
    version = gmsh_version()
    if not version:
        print(f"make_mesh_3d: WARNING: the gmsh version is unknown; the meshes of the paper were made "
              f"with gmsh {GMSH_VERSION}, other versions give other meshes", file=sys.stderr)
    elif version != GMSH_VERSION:
        print(f"make_mesh_3d: WARNING: gmsh {version}; the meshes of the paper were made with gmsh "
              f"{GMSH_VERSION}, other versions give other meshes", file=sys.stderr)
    foam = os.environ.get("WM_PROJECT_VERSION", "")
    if foam != "v2606":
        print(f"make_mesh_3d: WARNING: OpenFOAM '{foam or 'unknown'}'; the meshes of the paper were made "
              "with OpenFOAM v2606", file=sys.stderr)


def write_mesh_md(case, n, gmsh_stats, sizes, patches, check_line, steps, foam_build):
    lc = TARGET[n]
    patch_rows = "\n".join(f"| `{name}` | {kind} | {count} |" for name, kind, count in patches)
    step_rows = "\n".join(f"| {s['step']} | `{s['command']}` | {s['wall_s']:.2f} | {s['peak_rss_mib']:.1f} |"
                          for s in steps)
    n_bnd = sizes["nFaces"] - sizes["nInternalFaces"]
    how = f"= 0.19 / {n}" if float(lc) == 0.19 / n else f"0.19 / {n} rounded to 15 significant digits"
    g = {k: gmsh_stats.get(k, "?") for k in ("gmsh", "vertices", "tetrahedra", "boundary_triangles")}
    text = f"""# Example 2 mesh, level N = {n}

Polyhedral mesh of the box (-1.25, 1.25)^3 for Example 2, the 3D convergence test of the paper (the
convergence test in three dimensions).  All boundary faces form the patch
`boundary`.

| quantity | value |
|---|---|
| target edge length lc of the tetrahedral mesh | {lc} ({how}) |
| cells (= vertices of the tetrahedral mesh) | {sizes['nCells']} |
| faces: internal + boundary | {sizes['nFaces']} = {sizes['nInternalFaces']} + {n_bnd} |
| points | {sizes['nPoints']} |

| patch | type | faces |
|---|---|---|
{patch_rows}

## Construction

1. gmsh {g['gmsh']}, one thread: tetrahedral mesh of the box with target edge
   length lc = {lc} (Mesh.MeshSizeMin = Mesh.MeshSizeMax = lc); Frontal-Delaunay
   triangles on the sides (Mesh.Algorithm 6), Delaunay tetrahedra inside
   (Mesh.Algorithm3D 1), Mesh.Optimize 1, Mesh.OptimizeNetgen 1:
   {g['vertices']} vertices, {g['tetrahedra']} tetrahedra, {g['boundary_triangles']} boundary triangles (MSH 2.2 ascii,
   removed after step 2).
2. OpenFOAM `gmshToFoam`.
3. OpenFOAM `polyDualMesh {FEATURE_ANGLE} -overwrite`: one polyhedral cell per vertex of
   the tetrahedral mesh; the edges and corners of the box are kept.
4. OpenFOAM `combinePatchFaces {MERGE_ANGLE} -overwrite`: the boundary faces of a cell on
   one side of the box become one face.
5. OpenFOAM `renumberMesh -overwrite` (Cuthill-McKee order of the cells).
   OpenFOAM build {foam_build}.
6. Corner, edge and side rules (`check_mesh.py`): each corner of the box lies
   in one cell, a cell touches a box edge only along a segment, each box edge
   is split into segments of single cells and neighbouring cells on it share
   a face, and a cell has at most one boundary face per side of the box.

```
{check_line}
```

## Steps

Each step runs in its own process; peak memory is the maximum resident set
size of that process.  The logs are in `logs/`.

| step | command | wall time [s] | peak memory [MiB] |
|---|---|---|---|
{step_rows}

Made on {datetime.date.today().isoformat()} by `mesh/make_mesh_3d.py`.
"""
    with open(os.path.join(case, "MESH.md"), "w") as fh:
        fh.write(text)


def start_level(case):
    """Create the work directory N<N>.partial of a level.  A work directory left by an
    earlier failed or interrupted run of this script (it holds PARTIAL_MARK) is removed first."""
    if os.path.lexists(case):
        if not (os.path.isdir(case) and not os.path.islink(case)
                and os.path.isfile(os.path.join(case, PARTIAL_MARK))):
            raise StepFailed(f"{case} exists but was not made by this script; remove it or "
                             "choose another --out")
        print(f"make_mesh_3d: removing the incomplete level left by an earlier run: {case}")
        shutil.rmtree(case)
    os.makedirs(os.path.join(case, "logs"))
    with open(os.path.join(case, PARTIAL_MARK), "w") as fh:
        fh.write(PARTIAL_TEXT)


def finish_level(case, final):
    """Rename the complete work directory to the level directory N<N>."""
    if os.path.lexists(final):
        raise StepFailed(f"{final} appeared while the level was made; the new mesh stays in {case}")
    os.remove(os.path.join(case, PARTIAL_MARK))
    os.rename(case, final)


def make_level(root, n, figures):
    final = os.path.abspath(os.path.join(root, "example2", f"N{n}"))
    case = final + PARTIAL_SUFFIX                       # renamed to final after the last step
    start_level(case)
    logs = os.path.join(case, "logs")
    shutil.copytree(os.path.join(HERE, "system"), os.path.join(case, "system"))
    os.makedirs(os.path.join(case, "constant"))
    steps, lc = [], TARGET[n]
    started = time.perf_counter()
    msh = os.path.join(logs, "tetrahedra.msh")
    me = os.path.abspath(__file__)

    run_step("gmsh", [sys.executable, me, "--tetrahedra", lc, msh], case, os.path.join(logs, "log.gmsh"), steps)
    run_step("gmshToFoam", ["gmshToFoam", "-case", case, msh], case, os.path.join(logs, "log.gmshToFoam"),
             steps, cwd=logs)
    os.remove(msh)
    run_step("polyDualMesh", ["polyDualMesh", "-case", case, FEATURE_ANGLE, "-overwrite"], case,
             os.path.join(logs, "log.polyDualMesh"), steps, cwd=logs)
    for side_file in glob.glob(os.path.join(logs, "*.obj")):      # feature edges/points written by polyDualMesh
        os.remove(side_file)
    run_step("combinePatchFaces", ["combinePatchFaces", "-case", case, MERGE_ANGLE, "-overwrite"], case,
             os.path.join(logs, "log.combinePatchFaces"), steps, cwd=logs)
    run_step("renumberMesh", ["renumberMesh", "-case", case, "-overwrite"], case,
             os.path.join(logs, "log.renumberMesh"), steps, cwd=logs)
    shutil.rmtree(os.path.join(case, "constant", "polyMesh", "sets"), ignore_errors=True)
    check_log = os.path.join(logs, "log.check_mesh")
    try:
        run_step("check_mesh", [sys.executable, os.path.join(HERE, "check_mesh.py"), case, "--reject", "-v"],
                 case, check_log, steps)
    except StepFailed:
        with open(check_log) as fh:
            sys.stderr.write(fh.read())
        raise StepFailed("the mesh breaks the corner/edge/side rules and was moved to "
                         f"constant/polyMesh.rejected; see {check_log}")
    if figures:
        run_step("plot_mesh", [sys.executable, os.path.join(HERE, "plot_mesh.py"), case,
                               "--title", f"Example 2 mesh N = {n}"], case, os.path.join(logs, "log.plot_mesh"), steps)

    sizes, patches = mesh_sizes(case)
    gmsh_stats = stats_of(os.path.join(logs, "log.gmsh"))
    check_line = last_line(check_log)
    write_mesh_md(case, n, gmsh_stats, sizes, patches, check_line, steps,
                  openfoam_build(os.path.join(logs, "log.renumberMesh")))
    with open(os.path.join(logs, "steps.json"), "w") as fh:
        json.dump({"test": "example2", "level": n, "lc": lc, "steps": steps}, fh, indent=1)
    finish_level(case, final)
    print(f"example2/N{n}: {sizes['nCells']} cells, {sizes['nFaces']} faces, {sizes['nPoints']} points; "
          f"{check_line.split('|')[0].strip()}; {time.perf_counter() - started:.1f} s")


def remaining_command(levels, root):
    """The command that makes the given levels in root."""
    cmd = "python mesh/make_mesh_3d.py --levels " + " ".join(map(str, levels))
    if os.path.realpath(root) != os.path.realpath(os.path.join(REPO, "meshes")):
        cmd += " --out " + shlex.quote(root)
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the polyhedral meshes of Example 2 (the 3D convergence test of the paper).")
    ap.add_argument("--out", default=os.path.join(REPO, "meshes"),
                    help="mesh root directory; level N goes to OUT/example2/N<N> (default: %(default)s)")
    ap.add_argument("--levels", type=int, nargs="+", choices=sorted(TARGET), default=sorted(TARGET),
                    help="levels N to generate (default: 1 2 3 4)")
    ap.add_argument("--figures", action="store_true", help="also draw figures (needs matplotlib)")
    ap.add_argument("--tetrahedra", nargs=2, metavar=("LC", "MSH"), help=argparse.SUPPRESS)
    ap.add_argument("--gmsh-version", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.tetrahedra:                                # internal: step 1
        return tetrahedra(*args.tetrahedra)
    if args.gmsh_version:                              # internal: import check of gmsh
        import gmsh
        print(getattr(gmsh, "__version__", ""))
        return 0

    levels = sorted(set(args.levels))
    root = os.path.abspath(args.out)
    existing = [os.path.join(root, "example2", f"N{n}") for n in levels
                if os.path.lexists(os.path.join(root, "example2", f"N{n}"))]
    if existing:
        print("make_mesh_3d: these level directories exist already and are left untouched "
              "(remove them to make the levels again):\n  " + "\n  ".join(existing), file=sys.stderr)
        return 1
    check_prerequisites(args.figures)
    for k, n in enumerate(levels):
        partial = os.path.join(root, "example2", f"N{n}") + PARTIAL_SUFFIX
        try:
            make_level(root, n, args.figures)
        except (StepFailed, KeyboardInterrupt) as err:
            what = "interrupted" if isinstance(err, KeyboardInterrupt) else str(err)
            msg = [f"make_mesh_3d: example2/N{n}: {what}"]
            if os.path.isfile(os.path.join(partial, PARTIAL_MARK)):
                msg.append(f"  The incomplete level is kept in {partial} (logs/ holds the log of "
                           "every step); the next run of this script replaces it.")
            if k:
                msg.append("  Levels made: " + " ".join(f"N{m}" for m in levels[:k]) + ".")
            msg.append(f"  To make the remaining levels: {remaining_command(levels[k:], root)}")
            print("\n".join(msg), file=sys.stderr)
            return 130 if isinstance(err, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
