#!/usr/bin/env python3
# Copyright (C) 2026 Hyeokjoo Park and Jooyoung Hahn
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Licensed under the GNU General Public License v3.0 or later; see the file LICENSE.
"""Figures of a mesh made by make_mesh_2d.py or make_mesh_3d.py.

usage: python mesh/plot_mesh.py CASE [--out DIR] [--title TEXT] [--y0 Y]

Two-dimensional mesh (an OpenFOAM slab with an 'empty' patch):
  mesh.png         the polygonal mesh; corner, side and interior cells coloured;
  mesh_corner.png  the corner (-1.25, -1.25) of the square with the cell vertices
                   and, when CASE/logs/triangulation.npz exists, the triangles.
Three-dimensional mesh:
  mesh_half.png    (a) the cells with centre y < Y, whole cells (not clipped),
                   (b) the exact section of every cell by the plane y = Y;
  mesh_cells.png   a few interior cells;
  mesh_corner.png  the boundary faces near the corner (1.25, 1.25, 1.25): the
                   corner cell, the edge cells (two alternating colours along
                   each box edge, so that every segment is visible) and the
                   side cells.
The figures go to CASE/figures unless --out is given.  Needs matplotlib.  The
figures are meant for the coarse levels; for the fine 3D levels they take a
few minutes.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.legend_handler import HandlerTuple                 # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection            # noqa: E402

from check_mesh import analyse_box, edge_chains, read_polymesh     # noqa: E402

RED, BLUE, LIGHT_BLUE, GREY, LIGHT_GREY = "#d62728", "#1f5fbf", "#8fb3ea", "#bdbdbd", "#e6e6e6"


def shoelace(P):
    return 0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


# ---------------------------------------------------------------------------
# Two dimensions
# ---------------------------------------------------------------------------
def plot_2d(mesh, box, case, out, title):
    k0 = box["empty_axis"]
    plane = [a for a in range(3) if a != k0]
    fs = box["face_side"]
    polygons = [None] * mesh.n_cells
    n_side_faces = np.zeros(mesh.n_cells, dtype=int)
    for f in range(mesh.n_internal, mesh.n_faces):
        if fs[f] == 2 * k0:                                   # the cell polygon on the plane z = min
            polygons[mesh.owner[f]] = mesh.points[mesh.face(f)][:, plane]
        elif fs[f] >= 0 and fs[f] // 2 != k0:
            n_side_faces[mesh.owner[f]] += 1
    kind = np.minimum(n_side_faces, 2)                        # 0 interior, 1 side, 2 corner
    colour = {2: RED, 1: BLUE, 0: LIGHT_GREY}
    label = {2: "corner cells", 1: "side cells", 0: "interior cells"}
    lo, hi = box["lo"][plane], box["hi"][plane]
    many = mesh.n_cells > 2000

    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    ax.add_collection(PolyCollection(polygons, facecolors=[colour[k] for k in kind], edgecolors="k",
                                     linewidths=0.15 if many else 0.6))
    pad = 0.02 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(handles=[Patch(fc=colour[k], ec="k", label=f"{label[k]} ({int((kind == k).sum())})")
                       for k in (2, 1, 0)], loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              frameon=False)
    ax.set_title(f"{title}: {mesh.n_cells} polygonal cells", fontsize=11)
    fig.savefig(os.path.join(out, "mesh.png"), dpi=200 if many else 150, bbox_inches="tight")
    plt.close(fig)

    # the corner (min, min) of the square
    h = float(np.sqrt(np.prod(hi - lo) / mesh.n_cells))       # typical cell size
    x0, y0 = lo
    w = 5.0 * h
    window = (x0 - 0.15 * h, x0 + w), (y0 - 0.15 * h, y0 + w)
    shown = [c for c, P in enumerate(polygons) if P[:, 0].min() < x0 + w and P[:, 1].min() < y0 + w]
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    handles = []
    tri_file = os.path.join(case, "logs", "triangulation.npz")
    if os.path.exists(tri_file):
        data = np.load(tri_file)
        X, T = data["points"], data["triangles"]
        edges = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
        near = [X[e] for e in edges if X[e][:, 0].min() < x0 + w + h and X[e][:, 1].min() < y0 + w + h]
        ax.add_collection(LineCollection(near, colors="0.4", linewidths=0.9, linestyles=(0, (4, 3)), zorder=1))
        inside = (X[:, 0] < x0 + w + h) & (X[:, 1] < y0 + w + h)
        ax.plot(X[inside, 0], X[inside, 1], "o", ms=6, mfc="white", mec="0.3", zorder=3)
        handles += [Line2D([], [], color="0.4", lw=0.9, ls=(0, (4, 3)), label="triangles (primal mesh)"),
                    Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec="0.3",
                           label="triangle vertex (one cell each)")]
    ax.add_collection(PolyCollection([polygons[c] for c in shown], facecolors=[colour[kind[c]] for c in shown],
                                     alpha=0.5, edgecolors="none", zorder=0))
    ax.add_collection(PolyCollection([polygons[c] for c in shown], facecolors="none", edgecolors="k",
                                     linewidths=1.5, zorder=2))
    vertices = np.unique(np.vstack([polygons[c] for c in shown]), axis=0)
    on = ((np.abs(vertices - lo) < 1e-9 * h) | (np.abs(vertices - hi) < 1e-9 * h)).sum(axis=1)
    styles = [(0, "o", 4.5, "k", "cell vertex: triangle centroid"),
              (1, "s", 6.0, "#1f4e9c", "cell vertex on a side: midpoint of a boundary edge"),
              (2, "*", 15.0, RED, "cell vertex: corner of the square")]
    for n_on, marker, size, col, text in styles:
        P = vertices[on == n_on]
        ax.plot(P[:, 0], P[:, 1], marker, ms=size, color=col, mec="k" if n_on == 2 else col, ls="", zorder=4)
        handles.append(Line2D([], [], marker=marker, ls="", ms=size, color=col, label=text))
    handles += [Line2D([], [], color="k", lw=1.5, label="polygonal cells (dual mesh)")]
    handles += [Patch(fc=colour[k], alpha=0.5, label=label[k]) for k in (2, 1, 0)]
    ax.set_xlim(*window[0])
    ax.set_ylim(*window[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=False)
    ax.set_title(f"{title}: corner ({x0:g}, {y0:g})\none cell at the corner, one boundary segment per cell and "
                 "side, neighbouring boundary cells share a facet", fontsize=10)
    fig.savefig(os.path.join(out, "mesh_corner.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ["mesh.png", "mesh_corner.png"]


# ---------------------------------------------------------------------------
# Three dimensions
# ---------------------------------------------------------------------------
def section_loops(mesh, faces, y0, above):
    """Closed loops (arrays of (x, z)) where the plane y = y0 cuts a cell with the given faces."""
    pts = mesh.points
    cut, links = {}, defaultdict(list)
    for f in faces:
        v = mesh.face(f).tolist()
        hits = []
        for i in range(len(v)):
            a, b = v[i], v[(i + 1) % len(v)]
            if above[a] != above[b]:
                key = (min(a, b), max(a, b))
                if key not in cut:
                    pa, pb = pts[a], pts[b]
                    cut[key] = pa + (y0 - pa[1]) / (pb[1] - pa[1]) * (pb - pa)
                hits.append(key)
        if len(hits) < 2:
            continue
        # the crossings of one (planar) face lie on a line; pair them along it
        P = np.array([cut[k][[0, 2]] for k in hits])
        along = int(np.argmax(P.max(axis=0) - P.min(axis=0)))
        order = sorted(range(len(hits)), key=lambda i: P[i, along])
        for i in range(0, len(order) - 1, 2):
            a, b = hits[order[i]], hits[order[i + 1]]
            links[a].append(b)
            links[b].append(a)
    loops, used = [], set()
    for start in links:
        if start in used:
            continue
        loop, prev, cur = [start], None, start
        used.add(start)
        while True:
            nxt = [k for k in links[cur] if k != prev]
            if not nxt or nxt[0] == start or nxt[0] in used:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            used.add(cur)
        if len(loop) >= 3:
            loops.append(np.array([cut[k][[0, 2]] for k in loop]))
    return loops


def cell_classes(mesh, box):
    """0 interior, 1 side, 2 edge, 3 corner; and the box edge chains."""
    kind = np.zeros(mesh.n_cells, dtype=int)
    kind[mesh.owner[mesh.n_internal:]] = 1
    chains = edge_chains(mesh, box)
    for chain in chains.values():
        for *_, cells in chain:
            kind[cells] = 2
    for p in np.flatnonzero(box["n_sides"] == 3).tolist():
        kind[list(box["point_cells"][p])] = 3
    return kind, chains


def plot_3d(mesh, box, out, title, y0):
    pts = mesh.points
    cell_faces = mesh.cell_faces()
    n_faces = np.array([len(cf) for cf in cell_faces])
    centre = np.array([pts[np.unique(np.concatenate([mesh.face(f) for f in cf]))].mean(axis=0)
                       for cf in cell_faces])
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(n_faces.min(), n_faces.max())
    lo, hi = box["lo"], box["hi"]
    nif = mesh.n_internal
    files = []

    # (a) whole cells with centre y < y0 and (b) the exact section y = y0
    keep = centre[:, 1] < y0
    cut_faces, cut_colours, outer = [], [], []
    for f in range(mesh.n_faces):
        o = mesh.owner[f]
        if f < nif:
            n = mesh.neighbour[f]
            if keep[o] != keep[n]:
                cut_faces.append(pts[mesh.face(f)])
                cut_colours.append(cmap(norm(n_faces[o if keep[o] else n])))
        elif keep[o]:
            outer.append(pts[mesh.face(f)])
    fig = plt.figure(figsize=(15, 7.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(outer, facecolor=LIGHT_GREY, edgecolor="0.35", linewidth=0.25))
    ax.add_collection3d(Poly3DCollection(cut_faces, facecolors=cut_colours, edgecolor="k", linewidth=0.35))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=22, azim=62)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"(a) cells with centre y < {y0:g}, whole cells kept\n{int(keep.sum())} of {mesh.n_cells} "
                 "cells; the cut surface consists of cell faces", fontsize=11)
    above = pts[:, 1] >= y0
    sections, colours, n_cut = [], [], 0
    for c, cf in enumerate(cell_faces):
        yv = pts[np.concatenate([mesh.face(f) for f in cf]), 1]
        if yv.min() < y0 <= yv.max():
            loops = section_loops(mesh, cf, y0, above)
            n_cut += bool(loops)
            sections += loops
            colours += [cmap(norm(n_faces[c]))] * len(loops)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.add_collection(PolyCollection(sections, facecolors=colours, edgecolors="k", linewidths=0.5))
    ax2.set_xlim(lo[0], hi[0])
    ax2.set_ylim(lo[2], hi[2])
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title(f"(b) exact section of every cell by the plane y = {y0:g}\n{n_cut} cells cut", fontsize=11)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax, ax2], shrink=0.75, pad=0.02)
    bar.set_label("number of faces of the cell")
    fig.suptitle(f"{title}: {mesh.n_cells} polyhedral cells", fontsize=13)
    fig.savefig(os.path.join(out, "mesh_half.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    files.append("mesh_half.png")

    # a few interior cells near the centre, with different numbers of faces
    kind, chains = cell_classes(mesh, box)
    mid = 0.5 * (lo + hi)
    dist = np.linalg.norm(centre - mid, axis=1)
    pool = np.flatnonzero(kind == 0)
    pool = pool[np.argsort(dist[pool], kind="stable")][:max(60, int(0.02 * len(pool)))]
    counts = np.unique(n_faces[pool])
    targets = counts[np.unique(np.linspace(0, len(counts) - 1, min(8, len(counts))).round().astype(int))]
    picks = [int(pool[np.flatnonzero(n_faces[pool] == t)[0]]) for t in targets]
    fig = plt.figure(figsize=(4.0 * min(4, len(picks)), 4.3 * ((len(picks) + 3) // 4)))
    for i, c in enumerate(picks):
        axc = fig.add_subplot((len(picks) + 3) // 4, min(4, len(picks)), i + 1, projection="3d")
        polys = [pts[mesh.face(f)] for f in cell_faces[c]]
        axc.add_collection3d(Poly3DCollection(polys, facecolor=cmap(norm(n_faces[c])), edgecolor="k",
                                              linewidth=0.8, alpha=0.9))
        allp = np.vstack(polys)
        m, r = 0.5 * (allp.max(axis=0) + allp.min(axis=0)), 0.55 * (allp.max(axis=0) - allp.min(axis=0)).max()
        axc.set_xlim(m[0] - r, m[0] + r)
        axc.set_ylim(m[1] - r, m[1] + r)
        axc.set_zlim(m[2] - r, m[2] + r)
        axc.set_box_aspect((1, 1, 1))
        axc.view_init(elev=20, azim=35)
        axc.set_axis_off()
        n_vert = len(np.unique(allp, axis=0))
        axc.set_title(f"cell {c}: {n_faces[c]} faces, {n_vert} vertices", fontsize=10)
    fig.suptitle(f"{title}: interior cells near the centre of the box", fontsize=13)
    fig.savefig(os.path.join(out, "mesh_cells.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append("mesh_cells.png")

    # boundary faces near the corner (max, max, max)
    parity = {}
    for chain in chains.values():
        run = [cells[0] for *_, cells in chain]
        order = [c for i, c in enumerate(run) if i == 0 or c != run[i - 1]]
        for i, c in enumerate(order):
            parity.setdefault(c, i % 2)
    h = float(np.cbrt(np.prod(hi - lo) / mesh.n_cells))
    w = min(0.5 * float((hi - lo).max()), 6.0 * h)
    polys, colours = [], []
    for f in range(nif, mesh.n_faces):
        P = pts[mesh.face(f)]
        if np.all(P.mean(axis=0) > hi - w):
            c = mesh.owner[f]
            colours.append({3: RED, 2: BLUE if parity.get(c, 0) else LIGHT_BLUE, 1: LIGHT_GREY}[kind[c]])
            polys.append(P)
    fig = plt.figure(figsize=(8.2, 8.0))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolor="k", linewidth=0.6))
    ax.set_xlim(hi[0] - w, hi[0])
    ax.set_ylim(hi[1] - w, hi[1])
    ax.set_zlim(hi[2] - w, hi[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.legend([Patch(fc=RED, ec="k"), (Patch(fc=BLUE, ec="k"), Patch(fc=LIGHT_BLUE, ec="k")),
                Patch(fc=LIGHT_GREY, ec="k")],
               [f"corner cells ({int((kind == 3).sum())} in the mesh)",
                f"edge cells ({int((kind == 2).sum())} in the mesh), two colours alternating along each box edge",
                "side cells"],
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=1, fontsize=9, frameon=False)
    ax.set_title(f"{title}: boundary faces near the corner ({hi[0]:g}, {hi[1]:g}, {hi[2]:g})\n"
                 "one cell per corner; each box edge is split into segments, one per edge cell;\n"
                 "neighbouring edge cells share a face", fontsize=10)
    fig.savefig(os.path.join(out, "mesh_corner.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append("mesh_corner.png")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figures of a mesh made by make_mesh_2d.py or make_mesh_3d.py.")
    ap.add_argument("case", help="OpenFOAM case directory (reads constant/polyMesh)")
    ap.add_argument("--out", help="directory for the figures (default: CASE/figures)")
    ap.add_argument("--title", help="title of the figures (default: the name of CASE)")
    ap.add_argument("--y0", type=float, default=0.0, help="3D: position of the cut plane y = Y0 (default 0)")
    args = ap.parse_args(argv)
    case = os.path.abspath(args.case)
    out = args.out or os.path.join(case, "figures")
    title = args.title or os.path.basename(case)
    os.makedirs(out, exist_ok=True)
    mesh = read_polymesh(case)
    box = analyse_box(mesh)
    if box["empty_axis"] is not None:
        files = plot_2d(mesh, box, case, out, title)
    else:
        files = plot_3d(mesh, box, out, title, args.y0)
    print(f"plot_mesh: {mesh.n_cells} cells; wrote " + ", ".join(os.path.join(out, f) for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
