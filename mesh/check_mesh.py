#!/usr/bin/env python3
# Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
"""Check how the cells of a box mesh meet the corners, edges and sides of the box.

usage: python mesh/check_mesh.py CASE [--reject] [-v]

CASE is an OpenFOAM case directory with an ascii mesh in CASE/constant/polyMesh.
The domain is the axis-parallel box spanned by the mesh points.  The rules are

  1. each corner of the box lies in exactly one cell;
  2. a cell that touches an edge of the box touches it along a segment: one of
     the cell's own edges lies on the box edge (a cell touching a box edge in a
     single vertex only is not allowed);
  3. each edge of the box is split into segments without gaps, every segment
     belongs to one cell, and two cells meeting on a box edge share a face;
  4. a cell has at most one boundary face on each side of the box, so a corner
     cell has 3 boundary faces, an edge cell 2 and a side cell 1.

A two-dimensional mesh is stored by OpenFOAM as a slab one cell thick whose two
planes z = const form a patch of type 'empty'.  The rules are then checked on
the slab, and they read: each corner of the square lies in exactly one cell; a
cell touches a side of the square only along a segment; consecutive cells on a
side share a face; a cell has at most one boundary face per side (a corner cell
has 2); and every cell has one face on each of the two planes z = const.

The last line printed is a one-line summary.  The exit status is 0 when all
rules hold and 1 otherwise; with --reject a failing mesh is moved from
CASE/constant/polyMesh to CASE/constant/polyMesh.rejected.  The functions
read_polymesh() and analyse_box() are also used by plot_mesh.py.
"""
import argparse
import itertools
import os
import re
import sys
from collections import defaultdict

import numpy as np

AXES = "xyz"


# ---------------------------------------------------------------------------
# Reading an ascii OpenFOAM polyMesh
# ---------------------------------------------------------------------------
class PolyMesh:
    """Points, faces, owner, neighbour and patches of an OpenFOAM polyMesh.

    The faces are stored in compressed rows: the vertices of face f are
    face_points[face_start[f]:face_start[f + 1]].
    """

    def __init__(self, points, face_start, face_points, owner, neighbour, patches):
        self.points = points
        self.face_start = face_start
        self.face_points = face_points
        self.owner = owner
        self.neighbour = neighbour
        self.patches = patches
        self.n_faces = len(face_start) - 1
        self.n_internal = len(neighbour)
        self.n_cells = int(max(owner.max(initial=-1), neighbour.max(initial=-1)) + 1)
        if len(owner) != self.n_faces:
            raise ValueError(f"{len(owner)} owners for {self.n_faces} faces")

    def face(self, f):
        return self.face_points[self.face_start[f]:self.face_start[f + 1]]

    def cell_faces(self):
        """Face indices of every cell (a list of integer arrays)."""
        cells = np.concatenate([self.owner, self.neighbour])
        faces = np.concatenate([np.arange(self.n_faces), np.arange(self.n_internal)])
        order = np.argsort(cells, kind="stable")
        bounds = np.searchsorted(cells[order], np.arange(self.n_cells + 1))
        return [faces[order[bounds[c]:bounds[c + 1]]] for c in range(self.n_cells)]

    def patch_of_face(self):
        """Patch index of every face (-1 for internal faces)."""
        out = np.full(self.n_faces, -1, dtype=np.int64)
        for i, p in enumerate(self.patches):
            out[p["start"]:p["start"] + p["n_faces"]] = i
        return out


def _file_body(path):
    """Text of an OpenFOAM file after its FoamFile header, without comments."""
    with open(path) as fh:
        text = fh.read()
    head = text.find("FoamFile")
    if head < 0:
        raise SystemExit(f"check_mesh: {path} is not an OpenFOAM file (no FoamFile header)")
    stop = text.find("}", head)
    fmt = re.search(r"format\s+(\w+)\s*;", text[head:stop])
    if fmt and fmt.group(1) != "ascii":
        raise SystemExit(f"check_mesh: {path} is written as {fmt.group(1)}; only ascii meshes "
                         "can be read (convert the mesh with foamFormatConvert)")
    body = re.sub(r"/\*.*?\*/", " ", text[stop + 1:], flags=re.S)
    return re.sub(r"//[^\n]*", " ", body)


def _list_content(path):
    """Size and content (between the outer parentheses) of a list file."""
    body = _file_body(path)
    m = re.match(r"\s*(\d+)\s*\(", body)
    if m is None:
        raise SystemExit(f"check_mesh: cannot read the list in {path}")
    return int(m.group(1)), body[m.end():body.rindex(")")]


def _read_points(path):
    n, content = _list_content(path)
    values = np.array(content.replace("(", " ").replace(")", " ").split(), dtype=float)
    if values.size != 3 * n:
        raise SystemExit(f"check_mesh: {path}: {values.size} numbers for {n} points")
    return values.reshape(n, 3)


def _read_labels(path):
    n, content = _list_content(path)
    values = np.array(content.split(), dtype=np.int64)
    if values.size != n:
        raise SystemExit(f"check_mesh: {path}: {values.size} labels, expected {n}")
    return values


def _read_faces(path):
    n, content = _list_content(path)
    # every face is written as "<size>(<vertices>)"; only the sizes precede "("
    sizes = np.array(re.findall(r"(\d+)\s*\(", content), dtype=np.int64)
    vertices = np.array(re.sub(r"\d+\s*\(", " ", content).replace(")", " ").split(), dtype=np.int64)
    if sizes.size != n or vertices.size != sizes.sum():
        raise SystemExit(f"check_mesh: cannot read the {n} faces in {path}")
    return np.concatenate([[0], np.cumsum(sizes)]), vertices


def _read_patches(path):
    patches = []
    for m in re.finditer(r"([^\s{}()]+)\s*\{([^{}]*)\}", _file_body(path)):
        entries = dict(re.findall(r"(\w+)\s+([^;]*);", m.group(2)))
        patches.append({"name": m.group(1), "type": entries.get("type", "patch"),
                        "n_faces": int(entries["nFaces"]), "start": int(entries["startFace"])})
    return patches


def read_polymesh(case):
    """Read CASE/constant/polyMesh (ascii) into a PolyMesh."""
    d = os.path.join(case, "constant", "polyMesh")
    if not os.path.isdir(d):
        raise SystemExit(f"check_mesh: no mesh in {d}")
    start, vertices = _read_faces(os.path.join(d, "faces"))
    return PolyMesh(_read_points(os.path.join(d, "points")), start, vertices,
                    _read_labels(os.path.join(d, "owner")), _read_labels(os.path.join(d, "neighbour")),
                    _read_patches(os.path.join(d, "boundary")))


# ---------------------------------------------------------------------------
# Corners, edges and sides of the box
# ---------------------------------------------------------------------------
def side_name(s):
    """Sides of the box are numbered 2k (x_k = min) and 2k + 1 (x_k = max)."""
    return AXES[s // 2] + "-+"[s % 2]


def edge_name(e):
    return side_name(e[0]) + side_name(e[1])


def analyse_box(mesh):
    """Where the mesh meets the corners, edges and sides of its bounding box.

    A box edge is a pair (s, t) of sides on different axes, s < t.  Returns a
    dict with the per-point side flags, the cells and faces at every point on
    a box edge, the mesh edges lying on every box edge with the cells that
    contain them, and the side of every boundary face.
    """
    pts = mesh.points
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = hi - lo
    tol = 1e-9 * float(size.max())
    on = np.zeros((len(pts), 6), dtype=bool)
    for k in range(3):
        on[:, 2 * k] = np.abs(pts[:, k] - lo[k]) < tol
        on[:, 2 * k + 1] = np.abs(pts[:, k] - hi[k]) < tol
    n_sides = on.sum(axis=1)
    on_edge = n_sides >= 2                      # points on a box edge, corners included
    box_edges = [(s, t) for s, t in itertools.combinations(range(6), 2) if s // 2 != t // 2]

    nif, start, fp = mesh.n_internal, mesh.face_start, mesh.face_points
    face_of_entry = np.repeat(np.arange(mesh.n_faces), np.diff(start))

    # cells and faces around the points on box edges
    point_cells, point_faces = defaultdict(set), defaultdict(list)
    sel = on_edge[fp]
    for p, f in zip(fp[sel].tolist(), face_of_entry[sel].tolist()):
        point_faces[p].append(f)
        point_cells[p].add(int(mesh.owner[f]))
        if f < nif:
            point_cells[p].add(int(mesh.neighbour[f]))

    # mesh edges (consecutive face vertices) lying on a box edge, with their cells
    following = np.arange(len(fp)) + 1
    following[start[1:] - 1] = start[:-1]
    a_all, b_all = fp, fp[following]
    cand = on_edge[a_all] & on_edge[b_all]
    segment_cells = defaultdict(set)            # (box edge, a, b) -> cells, a < b
    degenerate = 0
    for a, b, f in zip(a_all[cand].tolist(), b_all[cand].tolist(), face_of_entry[cand].tolist()):
        common = np.flatnonzero(on[a] & on[b])
        if len(common) < 2:
            continue                            # a chord across a side, not along an edge
        if len(common) > 2:
            degenerate += 1                     # both ends at the same box corner
            continue
        key = ((int(common[0]), int(common[1])), min(a, b), max(a, b))
        segment_cells[key].add(int(mesh.owner[f]))
        if f < nif:
            segment_cells[key].add(int(mesh.neighbour[f]))

    # the side of every boundary face (-1: not on a side, -2: on more than one)
    face_side = np.full(mesh.n_faces, -1, dtype=np.int64)
    for f in range(nif, mesh.n_faces):
        sides = np.flatnonzero(on[fp[start[f]:start[f + 1]]].all(axis=0))
        face_side[f] = sides[0] if len(sides) == 1 else (-1 if len(sides) == 0 else -2)

    # a two-dimensional mesh: the 'empty' faces all lie on the two sides of one axis
    empty_axis = None
    empty_faces = [f for p in mesh.patches if p["type"] == "empty"
                   for f in range(p["start"], p["start"] + p["n_faces"])]
    if empty_faces:
        axes = set((face_side[empty_faces] // 2).tolist())
        if len(axes) == 1 and min(axes) >= 0:
            empty_axis = axes.pop()

    return {"lo": lo, "hi": hi, "size": size, "on": on, "n_sides": n_sides, "box_edges": box_edges,
            "point_cells": point_cells, "point_faces": point_faces, "segment_cells": segment_cells,
            "degenerate_segments": degenerate, "face_side": face_side, "empty_axis": empty_axis}


def edge_chains(mesh, box):
    """Mesh edges on every box edge, sorted along it: {box edge: [(t0, t1, p0, p1, cells)]}."""
    pts = mesh.points
    chains = defaultdict(list)
    for (e, a, b), cells in box["segment_cells"].items():
        free = 3 - e[0] // 2 - e[1] // 2
        if pts[a, free] > pts[b, free]:
            a, b = b, a
        chains[e].append((float(pts[a, free]), float(pts[b, free]), a, b, sorted(cells)))
    for e in chains:
        chains[e].sort()
    return chains


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
class Problems:
    """Rule violations, counted per kind, with a few examples of each."""

    def __init__(self, examples=3):
        self.count, self.example, self.examples = {}, {}, examples

    def add(self, kind, example=None):
        self.count[kind] = self.count.get(kind, 0) + 1
        if example is not None and len(self.example.setdefault(kind, [])) < self.examples:
            self.example[kind].append(example)

    def lines(self):
        out = []
        for kind, n in self.count.items():
            ex = self.example.get(kind, [])
            more = ", ..." if n > len(ex) and ex else ""
            out.append(f"{kind}: {n}" + (f" (e.g. {'; '.join(ex)}{more})" if ex else ""))
        return out

    def __bool__(self):
        return bool(self.count)


def check(mesh, box):
    """Check rules 1-4; return (problem lines, statistics, one-line summary)."""
    bad = Problems()
    on, n_sides, pc = box["on"], box["n_sides"], box["point_cells"]
    k0 = box["empty_axis"]
    two_d = k0 is not None
    if box["degenerate_segments"]:
        bad.add("face edges joining a box corner to itself")

    # rule 1: one cell per box corner
    corners = np.flatnonzero(n_sides == 3)
    corner_cells = [len(pc[int(p)]) for p in corners]
    if len(corners) != 8:
        bad.add(f"rule 1, mesh points at the 8 box corners: {len(corners)} found")
    for p, n in zip(corners.tolist(), corner_cells):
        if n != 1:
            bad.add("rule 1, box corners in more than one cell", f"point {p} in {n} cells")

    # rule 2: every (cell, box edge) contact is along a segment of the cell
    has_segment = {(c, e) for (e, _, _), cells in box["segment_cells"].items() for c in cells}
    contacts = set()
    for p in np.flatnonzero(n_sides >= 2).tolist():
        for e in itertools.combinations(np.flatnonzero(on[p]).tolist(), 2):
            for c in pc[p]:
                if (c, e) not in has_segment:
                    contacts.add((c, e))
    for c, e in sorted(contacts):
        bad.add("rule 2, cells touching a box edge in vertices only", f"cell {c} on {edge_name(e)}")

    # rule 3: each box edge split into segments of single cells without gaps;
    # neighbouring cells on a box edge share a face through their common point
    chains = edge_chains(mesh, box)
    lo, hi = box["lo"], box["hi"]
    edge_stats = {}
    for e in box["box_edges"]:
        free = 3 - e[0] // 2 - e[1] // 2
        chain = chains.get(e, [])
        name = edge_name(e)
        if not chain:
            bad.add("rule 3, box edges without mesh edges", name)
            continue
        length = sum(t1 - t0 for t0, t1, _, _, _ in chain)
        for t0, t1, _, _, cells in chain:
            if len(cells) != 1:
                bad.add("rule 3, segments in more than one cell", f"{name} at {t0:.6g}..{t1:.6g}: cells {cells}")
        gaps = sum(1 for s, t in zip(chain, chain[1:]) if s[3] != t[2])
        if gaps or n_sides[chain[0][2]] != 3 or n_sides[chain[-1][3]] != 3:
            bad.add("rule 3, box edges not covered by one chain of segments from corner to corner",
                    f"{name}: {gaps} gaps or overlaps")
        if abs(length - (hi[free] - lo[free])) > 1e-9 * (hi[free] - lo[free]):
            bad.add("rule 3, box edges whose segments do not add up to the edge length",
                    f"{name}: {length:.12g} of {hi[free] - lo[free]:.12g}")
        owners = [cells[0] for _, _, _, _, cells in chain]
        runs = [c for i, c in enumerate(owners) if i == 0 or c != owners[i - 1]]
        if len(runs) != len(set(runs)):
            bad.add("rule 3, cells with separate pieces on one box edge", name)
        junctions = 0
        for s, t in zip(chain, chain[1:]):
            p, a, b = s[3], s[4][0], t[4][0]
            if a == b:
                if pc[p] != {a}:
                    bad.add("rule 3, points inside a segment that belong to other cells too",
                            f"point {p} on {name}, cells {sorted(pc[p])}")
                continue
            if pc[p] == {a, b} and any(f < mesh.n_internal and {int(mesh.owner[f]), int(mesh.neighbour[f])} == {a, b}
                                       for f in box["point_faces"][p]):
                junctions += 1
            else:
                bad.add("rule 3, neighbouring cells on a box edge without a shared face",
                        f"cells {a}, {b} at point {p} on {name}")
        edge_stats[e] = {"segments": len(chain), "cells": len(runs), "junctions": junctions, "length": length}

    # rule 4: at most one boundary face per cell and side of the box
    fs = box["face_side"]
    bnd = np.arange(mesh.n_internal, mesh.n_faces)
    for f in bnd[fs[bnd] < 0].tolist():
        bad.add("rule 4, boundary faces not on one side of the box", f"face {f}")
    good = bnd[fs[bnd] >= 0]
    pairs, counts = np.unique(np.stack([mesh.owner[good], fs[good]], axis=1), axis=0, return_counts=True)
    for (c, sd), n in zip(pairs.tolist(), counts.tolist()):
        if n > 1:
            bad.add("rule 4, cells with more than one boundary face on a side", f"cell {c}: {n} on {side_name(sd)}")
    multi = int(np.count_nonzero(counts > 1))
    # boundary faces per cell on the sides of the domain (in 2D without the planes z = const)
    domain_side = (pairs[:, 1] // 2 != k0) if two_d else np.ones(len(pairs), dtype=bool)
    per_cell = np.bincount(pairs[domain_side, 0], weights=counts[domain_side], minlength=mesh.n_cells).astype(int)
    hist = {int(k): int(v) for k, v in enumerate(np.bincount(per_cell)) if k > 0 and v}
    corner_cell_set = sorted({c for p in corners.tolist() for c in pc[int(p)]})
    want = 2 if two_d else 3
    for c in corner_cell_set:
        if per_cell[c] != want:
            bad.add(f"rule 4, corner cells without {want} boundary faces", f"cell {c}: {per_cell[c]}")
    if two_d:
        on_planes = np.bincount(pairs[~domain_side, 0], weights=counts[~domain_side],
                                minlength=mesh.n_cells).astype(int)
        for c in np.flatnonzero(on_planes != 2).tolist():
            bad.add("slab, cells without exactly one face on each plane z = const", f"cell {c}")

    # summary, in the words of the dimension
    def edge_sum(edges, key):
        return sum(edge_stats.get(e, {}).get(key, 0) for e in edges)

    if two_d:
        cells_per_corner = []
        for e in [e for e in box["box_edges"] if e[0] // 2 != k0 and e[1] // 2 != k0]:
            cs = set()
            for p in corners.tolist():
                if on[p, e[0]] and on[p, e[1]]:
                    cs |= pc[int(p)]
            cells_per_corner.append(len(cs))
        sides = [e for e in box["box_edges"] if 2 * k0 in e]          # the sides of the square at z = min
        length = sum(float(box["size"][3 - e[0] // 2 - e[1] // 2]) for e in sides)
        rule1 = f"{len(cells_per_corner)} corners in {'/'.join(map(str, cells_per_corner))} cell(s)"
        rule3 = (f"4 sides split into {edge_sum(sides, 'segments')} segments, length "
                 f"{edge_sum(sides, 'length'):.6f} of {length:.6f}, {edge_sum(sides, 'junctions')} junctions "
                 "with a shared face")
        kind = f"2D square (slab), {mesh.n_cells} cells"
    else:
        edges = box["box_edges"]
        length = float(sum(box["size"][3 - e[0] // 2 - e[1] // 2] for e in edges))
        rule1 = f"{len(corners)} corners in {'/'.join(map(str, corner_cells))} cell(s)"
        rule3 = (f"12 edges split into {edge_sum(edges, 'segments')} segments, length "
                 f"{edge_sum(edges, 'length'):.6f} of {length:.6f}, {edge_sum(edges, 'junctions')} junctions "
                 "with a shared face")
        kind = f"3D box, {mesh.n_cells} cells"
    rule4 = ("boundary faces per boundary cell " + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items())) +
             f"; (cell, side) pairs with more than one face: {multi}")
    summary = (f"check_mesh {'FAILED' if bad else 'OK'} | {kind} | rule 1: {rule1} | rule 2: {len(contacts)} "
               f"vertex-only contacts | rule 3: {rule3} | rule 4: {rule4}")
    stats = {"corner_cells": corner_cell_set, "edge_stats": edge_stats, "faces_per_boundary_cell": hist}
    return bad.lines(), stats, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the corner, edge and side rules of a box mesh.")
    ap.add_argument("case", help="OpenFOAM case directory (reads constant/polyMesh)")
    ap.add_argument("--reject", action="store_true",
                    help="if a rule fails, rename constant/polyMesh to constant/polyMesh.rejected")
    ap.add_argument("-v", "--verbose", action="store_true", help="print details before the summary")
    args = ap.parse_args(argv)

    mesh = read_polymesh(args.case)
    box = analyse_box(mesh)
    problems, stats, summary = check(mesh, box)
    if args.verbose:
        lo, hi = box["lo"], box["hi"]
        print(f"case {args.case}")
        print(f"  box [{lo[0]:.6g}, {hi[0]:.6g}] x [{lo[1]:.6g}, {hi[1]:.6g}] x [{lo[2]:.6g}, {hi[2]:.6g}], "
              f"{mesh.n_cells} cells, {mesh.n_faces} faces ({mesh.n_internal} internal), "
              f"{len(mesh.points)} points")
        print("  patches: " + ", ".join(f"{p['name']} ({p['type']}, {p['n_faces']} faces)"
                                        for p in mesh.patches))
        for e, s in sorted(stats["edge_stats"].items()):
            print(f"  box edge {edge_name(e)}: {s['segments']} segments, {s['cells']} cells, "
                  f"{s['junctions']} junctions, length {s['length']:.6f}")
        print(f"  corner cells: {stats['corner_cells']}")
    for pr in problems:
        print("  PROBLEM: " + pr)
    print(summary)
    sys.stdout.flush()
    if problems and args.reject:
        pm = os.path.join(args.case, "constant", "polyMesh")
        target = pm + ".rejected"
        if os.path.lexists(target):
            print(f"check_mesh: {target} exists; the failing mesh stays in {pm}", file=sys.stderr)
        else:
            os.rename(pm, target)
            print(f"check_mesh: mesh rejected, moved to {target}", file=sys.stderr)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
