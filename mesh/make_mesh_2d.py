#!/usr/bin/env python3
# Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
"""Generate the polygonal meshes of Example 1 (the 2D convergence test of the paper).

usage: python mesh/make_mesh_2d.py [--out DIR] [--levels 1 2 3 4] [--figures]

Level N is written to DIR/example1/N<N> (default DIR: meshes/ in the repository)
as an OpenFOAM case directory: constant/polyMesh, system/, MESH.md and logs/,
plus figures/ with --figures.  An existing level directory is never touched;
remove it to make that level again.  A level is built in DIR/example1/N<N>.partial
and renamed to N<N> only after its last step, so N<N> always holds a complete
mesh.  If a step fails or the script is interrupted, N<N>.partial stays for
inspection (logs/ has the log of every step) and the next run replaces it.

Construction of level N, with the target edge length lc = 0.19 / 2^(N-1):
  1. gmsh 4.15.2 with one thread: Delaunay triangulation (Mesh.Algorithm 5)
     of the square (-1.25, 1.25)^2 with target edge length lc;
  2. dual mesh: one polygonal cell per vertex of the triangulation.  Its
     vertices are the centroids of the triangles around the vertex, plus the
     midpoints of the boundary edges and the corner of the square at the
     boundary, so that the boundary facets of a cell on one side of the square
     form a single facet.  OpenFOAM stores it as a slab one cell thick
     (0 <= z <= lc) with the patches 'boundary' (the four sides, type patch)
     and 'frontAndBack' (the planes z = 0 and z = lc, type empty);
  3. OpenFOAM renumberMesh -overwrite (Cuthill-McKee order of the cells);
  4. check_mesh.py: corner and side rules; a failing mesh is moved to
     constant/polyMesh.rejected and the script stops with an error.

Run the script in a shell with the OpenFOAM v2606 environment (renumberMesh
on PATH) and with a Python that has the packages of requirements.txt.  Every
step runs in its own process; its log, wall time and peak memory are kept in
logs/ and in MESH.md.
"""
import argparse
import datetime
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
HALF = 1.25                         # the square is (-HALF, HALF)^2
# target edge length 0.19 / 2^(N-1), as the decimal strings used for the paper
TARGET = {1: "0.19", 2: "0.095", 3: "0.0475", 4: "0.02375"}
GMSH_VERSION = "4.15.2"
OPENFOAM_TOOLS = ["renumberMesh"]
# environment of the steps: serial numerics, no .pyc files written next to the scripts
SERIAL = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          "PYTHONDONTWRITEBYTECODE": "1"}
# a level is built in N<N>.partial, which holds this marker file until it is complete
PARTIAL_SUFFIX = ".partial"
PARTIAL_MARK = "INCOMPLETE.txt"
PARTIAL_TEXT = "Level under construction by mesh/make_mesh_2d.py; removed by its next run.\n"


# ---------------------------------------------------------------------------
# Step 1 (own process): triangulation of the square with gmsh
# ---------------------------------------------------------------------------
def triangulate(lc_text, npz_path):
    """Delaunay triangulation of the square; writes points (n, 2) and triangles (m, 3)."""
    import gmsh
    import numpy as np

    lc = float(lc_text)
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", 1)      # the mesh depends on the thread count
    gmsh.model.add("square")
    gmsh.model.occ.addRectangle(-HALF, -HALF, 0.0, 2 * HALF, 2 * HALF)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)       # uniform target size lc
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)          # Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)

    # vertices sorted by node tag, triangles sorted by element tag
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    order = np.argsort(tags, kind="stable")
    tags, xyz = tags[order], xyz[order]
    types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2)
    if list(types) != [2]:
        raise SystemExit(f"triangulate: expected 3-node triangles only, got element types {list(types)}")
    elem_tags = np.asarray(elem_tags[0], dtype=np.int64)
    tri_tags = np.asarray(elem_nodes[0], dtype=np.int64).reshape(-1, 3)[np.argsort(elem_tags, kind="stable")]
    used = np.unique(tri_tags)
    dropped = len(tags) - len(used)
    points = xyz[np.searchsorted(tags, used), :2].copy()
    triangles = np.searchsorted(used, tri_tags)
    if np.abs(xyz[:, 2]).max() > 0:
        raise SystemExit("triangulate: vertices off the plane z = 0")

    # vertices within round-off of a side are put exactly on it
    snapped = 0
    for k in range(2):
        for side in (-HALF, HALF):
            near = np.abs(points[:, k] - side) < 1e-10 * HALF
            snapped += int(np.count_nonzero(points[near, k] != side))
            points[near, k] = side

    np.savez(npz_path, points=points, triangles=triangles)
    stats = {"gmsh": gmsh.__version__, "lc": lc_text, "vertices": len(points), "triangles": len(triangles),
             "unused_nodes": dropped, "snapped": snapped}
    gmsh.finalize()
    print("STATS " + json.dumps(stats))
    return 0


# ---------------------------------------------------------------------------
# Step 2 (own process): dual polygonal mesh, written as an OpenFOAM polyMesh
# ---------------------------------------------------------------------------
def build_dual(X, T):
    """Polygonal dual of the triangulation (X, T) of the square.

    Returns
      dual_points  the cell vertices: triangle centroids (one per triangle, in
                   triangle order), midpoints of the boundary edges (sorted by
                   edge) and the four corners of the square (sorted by y, x);
      polygons     per triangle vertex v, the counter-clockwise list of dual
                   point indices of cell v;
      facets       (P, Q, owner, neighbour) with neighbour = -1 on the boundary;
                   the owner lies on the left of P -> Q;
      kind         per cell: 0 interior, 1 side, 2 corner;
      counts       sizes of the triangulation.
    """
    import numpy as np

    nv, nt = len(X), len(T)
    T = T.copy()
    v = X[T]
    signed_area = 0.5 * ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
                         - (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0]))
    if np.any(signed_area == 0):
        raise SystemExit("dual: degenerate triangle")
    flipped = signed_area < 0
    T[flipped] = T[flipped][:, ::-1]                    # all triangles counter-clockwise

    left_of = {}                                        # directed edge -> triangle on its left
    for t in range(nt):
        for i in range(3):
            left_of[(int(T[t, i]), int(T[t, (i + 1) % 3]))] = t
    bedges = sorted({(min(a, b), max(a, b)) for (a, b) in left_of if (b, a) not in left_of})
    bedge_id = {e: k for k, e in enumerate(bedges)}
    on_side = np.abs(np.abs(X) - HALF) == 0.0           # exact, the boundary vertices are snapped
    for p, q in bedges:
        if not (on_side[p] & on_side[q] & (np.sign(X[p]) == np.sign(X[q]))).any():
            raise SystemExit(f"dual: boundary edge {p}-{q} is not on a side of the square")
    corners = [int(c) for c in np.flatnonzero(on_side.all(axis=1))]
    if len(corners) != 4:
        raise SystemExit(f"dual: {len(corners)} triangle vertices at the corners of the square, expected 4")
    corners.sort(key=lambda c: (X[c, 1], X[c, 0]))
    corner_id = {c: k for k, c in enumerate(corners)}
    on_boundary = {p for e in bedges for p in e}

    ends = np.array(bedges)
    dual_points = np.vstack([X[T].mean(axis=1),
                             (X[ends[:, 0]] + X[ends[:, 1]]) / 2,
                             X[corners]])

    def midpoint(a, b):
        return nt + bedge_id[(min(a, b), max(a, b))]

    def corner_point(c):
        return nt + len(bedges) + corner_id[c]

    around = [[] for _ in range(nv)]                    # (triangle, local index), by triangle
    for t in range(nt):
        for i in range(3):
            around[int(T[t, i])].append((t, i))

    def fan(p):
        """Triangles around vertex p, counter-clockwise, as (t, a, b): p -> a is the
        edge where t is entered and p -> b the edge where it is left.  At the
        boundary the fan starts at a boundary edge."""
        t, i = around[p][0]
        if p in on_boundary:                            # go back to the boundary edge
            for _ in range(len(around[p]) + 1):
                prev = left_of.get((int(T[t, (i + 1) % 3]), p))
                if prev is None:
                    break
                t = prev
                i = int(np.flatnonzero(T[t] == p)[0])
        out, first = [], t
        while True:
            i = int(np.flatnonzero(T[t] == p)[0])
            out.append((t, int(T[t, (i + 1) % 3]), int(T[t, (i + 2) % 3])))
            nxt = left_of.get((p, int(T[t, (i + 2) % 3])))
            if nxt is None or nxt == first:
                break
            t = nxt
            if len(out) > len(around[p]):
                raise SystemExit(f"dual: the triangles around vertex {p} do not form a fan")
        if len(out) != len(around[p]):
            raise SystemExit(f"dual: fan of vertex {p} has {len(out)} of {len(around[p])} triangles")
        return out

    polygons, kind, facets, first_seen = [], np.zeros(nv, dtype=int), [], {}
    for p in range(nv):
        F = fan(p)
        sides = []                                      # (P, Q, cell across, or -1)
        if p not in on_boundary:
            polygon = [t for t, _, _ in F]
            for j in range(len(F)):
                sides.append((F[j][0], F[(j + 1) % len(F)][0], F[j][2]))
        else:
            a, b = F[0][1], F[-1][2]                    # the two boundary neighbours of p
            ma, mb = midpoint(p, a), midpoint(p, b)
            polygon = [ma] + [t for t, _, _ in F] + [mb]
            sides.append((ma, F[0][0], a))
            for j in range(len(F) - 1):
                sides.append((F[j][0], F[j + 1][0], F[j][2]))
            sides.append((F[-1][0], mb, b))
            if p in corner_id:                          # two boundary facets, through the corner
                kind[p] = 2
                polygon.append(corner_point(p))
                sides.append((mb, corner_point(p), -1))
                sides.append((corner_point(p), ma, -1))
            else:                                       # one boundary facet on the side of p
                kind[p] = 1
                same = on_side[p] & (np.abs(dual_points[ma]) == HALF) & (np.abs(dual_points[mb]) == HALF)
                if not same.any():
                    raise SystemExit(f"dual: boundary facet of vertex {p} is not on one side")
                sides.append((mb, ma, -1))
        polygons.append(polygon)
        for P, Q, w in sides:
            if w < 0:
                facets.append((P, Q, p, -1))
                continue
            key = (min(p, w), max(p, w))
            if key in first_seen:                       # the facet seen from the other cell
                if first_seen[key] != (Q, P):
                    raise SystemExit(f"dual: facet between cells {key} differs between the two cells")
                continue
            first_seen[key] = (P, Q)
            facets.append((P, Q, p, w) if p < w else (Q, P, w, p))
    counts = {"vertices": nv, "triangles": nt,
              "edges": len({(min(a, b), max(a, b)) for (a, b) in left_of}),
              "boundary_edges": len(bedges), "flipped_triangles": int(flipped.sum())}
    return dual_points, polygons, facets, kind, counts


def foam_header(cls, obj, note=None):
    lines = ["/*--------------------------------*- C++ -*----------------------------------*\\",
             "  Polygonal mesh, one cell thick, written by mesh/make_mesh_2d.py",
             "\\*---------------------------------------------------------------------------*/",
             "FoamFile", "{",
             "    version     2.0;",
             "    format      ascii;",
             '    arch        "LSB;label=32;scalar=64";',
             f"    class       {cls};"]
    if note:
        lines.append(f'    note        "{note}";')
    lines += ['    location    "constant/polyMesh";', f"    object      {obj};", "}", "", ""]
    return "\n".join(lines)


def write_polymesh(directory, dual_points, polygons, facets, thickness):
    """Write the dual mesh as an ascii polyMesh: a slab 0 <= z <= thickness.

    Points: the dual points at z = 0, then the same points at z = thickness.
    Faces: internal facets (sorted by owner, neighbour), boundary facets
    (sorted by owner, P, Q), then per cell its polygon at z = 0 (normal -z)
    and at z = thickness (normal +z).
    """
    import numpy as np

    n2, n_cells = len(dual_points), len(polygons)
    points = np.vstack([np.c_[dual_points, np.zeros(n2)], np.c_[dual_points, np.full(n2, thickness)]])
    internal = sorted([f for f in facets if f[3] >= 0], key=lambda f: (f[2], f[3]))
    boundary = sorted([f for f in facets if f[3] < 0], key=lambda f: (f[2], f[0], f[1]))
    faces, owner, neighbour = [], [], []
    for P, Q, o, n in internal:
        faces.append((P, Q, Q + n2, P + n2))
        owner.append(o)
        neighbour.append(n)
    for P, Q, o, _ in boundary:
        faces.append((P, Q, Q + n2, P + n2))
        owner.append(o)
    for c in range(n_cells):
        faces.append(tuple(polygons[c][::-1]))
        owner.append(c)
        faces.append(tuple(q + n2 for q in polygons[c]))
        owner.append(c)
    n_int, n_bnd = len(internal), len(boundary)

    os.makedirs(directory, exist_ok=True)
    note = f"nPoints:{len(points)}  nCells:{n_cells}  nFaces:{len(faces)}  nInternalFaces:{n_int}"
    with open(os.path.join(directory, "points"), "w") as fh:
        fh.write(foam_header("vectorField", "points"))
        fh.write(f"{len(points)}\n(\n")
        fh.writelines(f"({x!r} {y!r} {z!r})\n" for x, y, z in points.tolist())
        fh.write(")\n")
    with open(os.path.join(directory, "faces"), "w") as fh:
        fh.write(foam_header("faceList", "faces"))
        fh.write(f"{len(faces)}\n(\n")
        fh.writelines(f"{len(face)}({' '.join(map(str, face))})\n" for face in faces)
        fh.write(")\n")
    for name, labels in (("owner", owner), ("neighbour", neighbour)):
        with open(os.path.join(directory, name), "w") as fh:
            fh.write(foam_header("labelList", name, note))
            fh.write(f"{len(labels)}\n(\n")
            fh.writelines(f"{label}\n" for label in labels)
            fh.write(")\n")
    with open(os.path.join(directory, "boundary"), "w") as fh:
        fh.write(foam_header("polyBoundaryMesh", "boundary"))
        fh.write("2\n(\n"
                 f"    boundary\n    {{\n        type            patch;\n        nFaces          {n_bnd};\n"
                 f"        startFace       {n_int};\n    }}\n"
                 f"    frontAndBack\n    {{\n        type            empty;\n        inGroups        1(empty);\n"
                 f"        nFaces          {2 * n_cells};\n        startFace       {n_int + n_bnd};\n    }}\n)\n")
    return {"points": len(points), "faces": len(faces), "internal_faces": n_int,
            "boundary_faces": n_bnd, "frontAndBack_faces": 2 * n_cells}


def polygon_area(P):
    """Signed area of the polygon with vertices P (n, 2), positive if counter-clockwise."""
    import numpy as np

    return 0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


def dual_step(npz_path, lc_text, case):
    """Read the triangulation, build the dual mesh, check it, write the polyMesh."""
    import numpy as np

    data = np.load(npz_path)
    X, T = data["points"], data["triangles"]
    dual_points, polygons, facets, kind, counts = build_dual(X, T)
    areas = np.array([polygon_area(dual_points[p]) for p in polygons])
    if np.any(areas <= 0):
        raise SystemExit(f"dual: {int(np.count_nonzero(areas <= 0))} cells with area <= 0")
    if abs(areas.sum() - (2 * HALF) ** 2) > 1e-12:
        raise SystemExit(f"dual: the cell areas add up to {areas.sum():.15f}, not {(2 * HALF) ** 2}")
    if len(polygons) != counts["vertices"]:
        raise SystemExit("dual: number of cells differs from the number of triangle vertices")
    n_int = sum(1 for f in facets if f[3] >= 0)
    if n_int != counts["edges"]:
        raise SystemExit(f"dual: {n_int} internal facets for {counts['edges']} triangle edges")
    sizes = write_polymesh(os.path.join(case, "constant", "polyMesh"), dual_points, polygons, facets,
                           float(lc_text))
    stats = dict(counts, cells=len(polygons), dual_points=len(dual_points), numpy=np.__version__,
                 corner_cells=int((kind == 2).sum()), side_cells=int((kind == 1).sum()),
                 interior_cells=int((kind == 0).sum()), area_min=float(areas.min()),
                 area_max=float(areas.max()), area_sum=float(areas.sum()), **sizes)
    print(f"dual mesh: {stats['cells']} cells ({stats['corner_cells']} corner, {stats['side_cells']} side, "
          f"{stats['interior_cells']} interior), cell areas {areas.min():.6e} .. {areas.max():.6e}, "
          f"sum {areas.sum():.15f}")
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
        raise SystemExit(f"make_mesh_2d: 'import gmsh' fails for {sys.executable}:\n  "
                         + (lines[-1] if lines else f"exit status {proc.returncode}")
                         + "\n  Install the Python packages (python -m pip install -r requirements.txt) and "
                         "the system libraries that the gmsh wheel needs (README.txt, Installation).")
    return proc.stdout.strip()


def check_prerequisites(figures):
    missing = [t for t in OPENFOAM_TOOLS if shutil.which(t) is None]
    if missing:
        raise SystemExit("make_mesh_2d: OpenFOAM utility not found on PATH: " + ", ".join(missing) +
                         "\n  Load the OpenFOAM v2606 environment first, as in step (0) of README.txt.")
    for module in ["numpy"] + (["matplotlib"] if figures else []):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"make_mesh_2d: the Python package '{module}' is missing for {sys.executable}"
                             "\n  Install the packages: python -m pip install -r requirements.txt")
    version = gmsh_version()
    if not version:
        print(f"make_mesh_2d: WARNING: the gmsh version is unknown; the meshes of the paper were made "
              f"with gmsh {GMSH_VERSION}, other versions give other meshes", file=sys.stderr)
    elif version != GMSH_VERSION:
        print(f"make_mesh_2d: WARNING: gmsh {version}; the meshes of the paper were made with gmsh "
              f"{GMSH_VERSION}, other versions give other meshes", file=sys.stderr)
    foam = os.environ.get("WM_PROJECT_VERSION", "")
    if foam != "v2606":
        print(f"make_mesh_2d: WARNING: OpenFOAM '{foam or 'unknown'}'; the meshes of the paper were made "
              "with OpenFOAM v2606", file=sys.stderr)


def write_mesh_md(case, n, gmsh_stats, dual_stats, sizes, patches, check_line, steps, foam_build):
    lc = TARGET[n]
    patch_rows = "\n".join(f"| `{name}` | {kind} | {count} |" for name, kind, count in patches)
    step_rows = "\n".join(f"| {s['step']} | `{s['command']}` | {s['wall_s']:.2f} | {s['peak_rss_mib']:.1f} |"
                          for s in steps)
    n_bnd = next((c for name, _, c in patches if name == "boundary"), 0)
    d = {k: dual_stats.get(k, "?") for k in ("corner_cells", "side_cells", "interior_cells")}
    text = f"""# Example 1 mesh, level N = {n}

Polygonal mesh of the square (-1.25, 1.25)^2 for Example 1, the 2D convergence test of the paper (the
convergence test in two dimensions).  OpenFOAM stores the two-dimensional mesh
as a slab one cell thick, 0 <= z <= {lc}: patch `boundary` holds the four sides
of the square and patch `frontAndBack` (type `empty`) the planes z = 0 and
z = {lc}.

| quantity | value |
|---|---|
| target edge length lc of the triangulation | {lc} (= 0.19 / 2^{n - 1}) |
| cells (= vertices of the triangulation) | {sizes['nCells']} |
| cells at a corner / on a side / interior | {d['corner_cells']} / {d['side_cells']} / {d['interior_cells']} |
| facets of the 2D mesh: internal + boundary | {sizes['nInternalFaces'] + n_bnd} = {sizes['nInternalFaces']} + {n_bnd} |
| OpenFOAM faces (incl. frontAndBack) | {sizes['nFaces']} |
| OpenFOAM points (both planes) | {sizes['nPoints']} |
| sum of the cell areas | {dual_stats['area_sum']:.15f} |

| patch | type | faces |
|---|---|---|
{patch_rows}

## Construction

1. gmsh {gmsh_stats.get('gmsh', '?')}, one thread: Delaunay triangulation (Mesh.Algorithm 5) of
   the square with target edge length lc = {lc} (Mesh.MeshSizeMin = Mesh.MeshSizeMax =
   lc, Mesh.Optimize 1): {gmsh_stats.get('vertices', '?')} vertices, {gmsh_stats.get('triangles', '?')} triangles
   (`logs/triangulation.npz`).
2. Dual mesh (`make_mesh_2d.py`, numpy {dual_stats.get('numpy', '?')}): one polygonal cell per
   vertex of the triangulation.  The cell vertices are the centroids of the
   triangles around the vertex and, at the boundary, the midpoints of the
   boundary edges and the corner of the square; the boundary facets of a cell
   on one side of the square form one facet.  Written as an ascii polyMesh.
3. OpenFOAM `renumberMesh -overwrite` (Cuthill-McKee order of the cells),
   OpenFOAM build {foam_build}.
4. Corner and side rules (`check_mesh.py`): each corner of the square lies in
   one cell, a cell touches a side only along a segment, consecutive cells on
   a side share a face, and a cell has at most one boundary face per side.

```
{check_line}
```

## Steps

Each step runs in its own process; peak memory is the maximum resident set
size of that process.  The logs are in `logs/`.

| step | command | wall time [s] | peak memory [MiB] |
|---|---|---|---|
{step_rows}

Made on {datetime.date.today().isoformat()} by `mesh/make_mesh_2d.py`.
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
        print(f"make_mesh_2d: removing the incomplete level left by an earlier run: {case}")
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
    final = os.path.abspath(os.path.join(root, "example1", f"N{n}"))
    case = final + PARTIAL_SUFFIX                       # renamed to final after the last step
    start_level(case)
    logs = os.path.join(case, "logs")
    shutil.copytree(os.path.join(HERE, "system"), os.path.join(case, "system"))
    os.makedirs(os.path.join(case, "constant"))
    steps, lc = [], TARGET[n]
    started = time.perf_counter()
    npz = os.path.join(logs, "triangulation.npz")
    me = os.path.abspath(__file__)

    run_step("gmsh", [sys.executable, me, "--triangulate", lc, npz], case, os.path.join(logs, "log.gmsh"), steps)
    run_step("dual", [sys.executable, me, "--dual", npz, lc, case], case, os.path.join(logs, "log.dual"), steps)
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
        raise StepFailed("the mesh breaks the corner/side rules and was moved to "
                         f"constant/polyMesh.rejected; see {check_log}")
    if figures:
        run_step("plot_mesh", [sys.executable, os.path.join(HERE, "plot_mesh.py"), case,
                               "--title", f"Example 1 mesh N = {n}"], case, os.path.join(logs, "log.plot_mesh"), steps)

    sizes, patches = mesh_sizes(case)
    gmsh_stats = stats_of(os.path.join(logs, "log.gmsh"))
    dual_stats = stats_of(os.path.join(logs, "log.dual"))
    check_line = last_line(check_log)
    write_mesh_md(case, n, gmsh_stats, dual_stats, sizes, patches, check_line, steps,
                  openfoam_build(os.path.join(logs, "log.renumberMesh")))
    with open(os.path.join(logs, "steps.json"), "w") as fh:
        json.dump({"test": "example1", "level": n, "lc": lc, "steps": steps}, fh, indent=1)
    finish_level(case, final)
    print(f"example1/N{n}: {sizes['nCells']} cells, {sizes['nFaces']} faces, {sizes['nPoints']} points; "
          f"{check_line.split('|')[0].strip()}; {time.perf_counter() - started:.1f} s")


def remaining_command(levels, root):
    """The command that makes the given levels in root."""
    cmd = "python mesh/make_mesh_2d.py --levels " + " ".join(map(str, levels))
    if os.path.realpath(root) != os.path.realpath(os.path.join(REPO, "meshes")):
        cmd += " --out " + shlex.quote(root)
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the polygonal meshes of Example 1 (the 2D convergence test of the paper).")
    ap.add_argument("--out", default=os.path.join(REPO, "meshes"),
                    help="mesh root directory; level N goes to OUT/example1/N<N> (default: %(default)s)")
    ap.add_argument("--levels", type=int, nargs="+", choices=sorted(TARGET), default=sorted(TARGET),
                    help="levels N to generate (default: 1 2 3 4)")
    ap.add_argument("--figures", action="store_true", help="also draw figures (needs matplotlib)")
    ap.add_argument("--triangulate", nargs=2, metavar=("LC", "NPZ"), help=argparse.SUPPRESS)
    ap.add_argument("--dual", nargs=3, metavar=("NPZ", "LC", "CASE"), help=argparse.SUPPRESS)
    ap.add_argument("--gmsh-version", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.triangulate:                               # internal: step 1
        return triangulate(*args.triangulate)
    if args.dual:                                      # internal: step 2
        return dual_step(*args.dual)
    if args.gmsh_version:                              # internal: import check of gmsh
        import gmsh
        print(getattr(gmsh, "__version__", ""))
        return 0

    levels = sorted(set(args.levels))
    root = os.path.abspath(args.out)
    existing = [os.path.join(root, "example1", f"N{n}") for n in levels
                if os.path.lexists(os.path.join(root, "example1", f"N{n}"))]
    if existing:
        print("make_mesh_2d: these level directories exist already and are left untouched "
              "(remove them to make the levels again):\n  " + "\n  ".join(existing), file=sys.stderr)
        return 1
    check_prerequisites(args.figures)
    for k, n in enumerate(levels):
        partial = os.path.join(root, "example1", f"N{n}") + PARTIAL_SUFFIX
        try:
            make_level(root, n, args.figures)
        except (StepFailed, KeyboardInterrupt) as err:
            what = "interrupted" if isinstance(err, KeyboardInterrupt) else str(err)
            msg = [f"make_mesh_2d: example1/N{n}: {what}"]
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
