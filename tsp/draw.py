"""
Matplotlib code for visualizing TSP problems and routes.

Adapted from code provided with the paper "Attention, Learn to Solve Routing Problems!"
https://github.com/wouterkool/attention-learn-to-route/blob/master/simple_tsp.ipynb
Liscense and copyright notice below.

MIT License

Copyright (c) 2018 Wouter Kool

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def plot_tsp(xy, tour, ax1, solution_tour=None, base_col="black", sol_col="green"):
    """
    Plot the TSP tour on matplotlib axis ax1.
    """
    xy = np.asarray(xy)
    tour = np.asarray(tour)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    xs, ys = np.split(xy[tour], 2, axis=-1)
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color="blue")
    # Starting node
    ax1.scatter([xs[0]], [ys[0]], s=100, color="red")

    # Add light green oracle trace in background, if provided
    if solution_tour is not None:
        xso, yso = np.split(xy[solution_tour], 2, axis=-1)  # 'o' for oracle
        dxo = np.roll(xso, -1) - xso
        dyo = np.roll(yso, -1) - yso
        do = np.sqrt(dxo * dxo + dyo * dyo)
        lengths_oracle = do.cumsum()

        ax1.quiver(
            xso,
            yso,
            dxo,
            dyo,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=0.001,
            color=sol_col
        )

        title_str = "{} nodes, total length {:.2f} ({:.2f})".format(len(tour), lengths[-1], lengths_oracle[-1])

    else:
        title_str = "{} nodes, total length {:.2f}".format(len(tour), lengths[-1])

    # Primary tour arcs
    ax1.quiver(
        xs,
        ys,
        dx,
        dy,
        scale_units="xy",
        angles="xy",
        scale=1,
        width=0.002,
        color=base_col
    )

    ax1.set_title(title_str)
