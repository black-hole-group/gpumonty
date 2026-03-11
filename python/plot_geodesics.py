#!/usr/bin/env python3
"""
Plot 3D photon geodesic trajectories from GPUmonty HDF5 output.

Usage:
    python plot_geodesics.py [geodesics.h5] [--n N] [--no-bh]

Arguments:
    geodesics.h5    Path to HDF5 file (default: output/geodesics.h5)
    --n N           Number of geodesics to plot (default: 50)
    --no-bh         Skip drawing the black hole sphere

What it does:
    Reads the r, theta, phi arrays from the HDF5 file produced by GPUmonty's
    geodesic tracing mode. Each photon's valid steps are determined by the
    nsteps array, so truncated trajectories (photons that fell through the
    horizon or escaped the outer boundary early) are handled correctly.
    Boyer-Lindquist coordinates are converted to Cartesian (x, y, z) in units
    of r_g. A random subset of --n geodesics is selected and drawn as thin
    colored lines in an interactive Plotly 3D scene. An opaque black sphere at
    r = 1 r_g marks the black hole horizon (disable with --no-bh).
"""

import argparse
import sys
import numpy as np
import h5py
import plotly.graph_objects as go


def bl_to_cartesian(r, theta, phi):
    """Convert Boyer-Lindquist spherical coords to Cartesian."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def load_geodesics(h5path, n_plot):
    with h5py.File(h5path, "r") as f:
        nph = int(f["nph"][()])
        r_all = f["r"][:]       # shape [nph, max_saved]
        th_all = f["theta"][:]
        ph_all = f["phi"][:]
        nsteps = f["nsteps"][:].astype(int)

        # metadata for title
        meta = {
            "nph": nph,
            "Ns": int(f["Ns"][()]) if "Ns" in f else None,
            "trace_stride": int(f["trace_stride"][()]) if "trace_stride" in f else None,
        }

    n_plot = min(n_plot, nph)
    idx = np.random.choice(nph, size=n_plot, replace=False)
    return r_all, th_all, ph_all, nsteps, idx, meta


def make_bh_sphere(r_horizon=1.0, n=40):
    """Return a Plotly surface trace for the black hole horizon sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = r_horizon * np.outer(np.cos(u), np.sin(v))
    y = r_horizon * np.outer(np.sin(u), np.sin(v))
    z = r_horizon * np.outer(np.ones(n), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "black"], [1, "black"]],
        showscale=False,
        opacity=1.0,
        name="Black hole horizon",
    )


def plot_geodesics(h5path, n_plot, show_bh):
    r_all, th_all, ph_all, nsteps, idx, meta = load_geodesics(h5path, n_plot)

    # Color scale: sequential colormap across geodesics
    colors = [f"hsl({int(360 * i / len(idx))},70%,55%)" for i in range(len(idx))]

    traces = []

    if show_bh:
        traces.append(make_bh_sphere(r_horizon=1.0))

    for color, i in zip(colors, idx):
        ns = nsteps[i]
        if ns == 0:
            continue
        r = r_all[i, :ns]
        th = th_all[i, :ns]
        ph = ph_all[i, :ns]
        x, y, z = bl_to_cartesian(r, th, ph)
        traces.append(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
                hovertemplate=(
                    "r=%{customdata[0]:.3f}<br>"
                    "θ=%{customdata[1]:.3f}<br>"
                    "φ=%{customdata[2]:.3f}<extra></extra>"
                ),
                customdata=np.stack([r, th, ph], axis=-1),
            )
        )

    title = f"Photon geodesics — {len(idx)} of {meta['nph']} traced"
    if meta["Ns"]:
        title += f"  (Ns={meta['Ns']})"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [r_g]",
            yaxis_title="y [r_g]",
            zaxis_title="z [r_g]",
            aspectmode="data",
            bgcolor="rgb(10,10,20)",
            xaxis=dict(gridcolor="gray", zerolinecolor="gray"),
            yaxis=dict(gridcolor="gray", zerolinecolor="gray"),
            zaxis=dict(gridcolor="gray", zerolinecolor="gray"),
        ),
        paper_bgcolor="rgb(20,20,30)",
        font=dict(color="white"),
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5file", nargs="?", default="output/geodesics.h5",
                        help="Path to HDF5 file (default: output/geodesics.h5)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of geodesics to plot (default: 50)")
    parser.add_argument("--no-bh", action="store_true",
                        help="Skip drawing the black hole horizon sphere")
    args = parser.parse_args()

    plot_geodesics(args.h5file, args.n, not args.no_bh)


if __name__ == "__main__":
    main()
