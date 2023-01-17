"""
Visualize logged "progress.csv" files with Visdom.
"""

from visdom import Visdom
import numpy as np
import os.path as osp
import time
import argparse

DEFAULT_X_KEY = "iteration"  # string identifying key for default x-axis in logging CSV


class LinePlotter:
    """
    Manages Visdom line window for single and repeated (appending)
    plotting of array data. Expects to be given a shared Visdom
    context 'viz'.
    """

    def __init__(self, viz, xlabel, ylabel, title_name, env_name="main"):
        self.viz = viz

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title_name = title_name

        self.env = env_name
        self.window = None

    def plot(self, x, y):
        if self.window is None:
            self.window = self.viz.line(
                X=np.asarray(x),
                Y=np.asarray(y),
                env=self.env,
                opts=dict(
                    title=self.title_name, xlabel=self.xlabel, ylabel=self.ylabel
                ),
            )
        else:
            self.viz.line(
                X=np.asarray(x),
                Y=np.asarray(y),
                env=self.env,
                win=self.window,
                update="append",
            )


class LogPlotter:
    """
    Visualize all lines in a CSV log file (e.g. progress.csv).

    Assumes CSV is rllab/viskit logging formatted,
    so the first line contains the (fixed) keys,
    and subsequent lines contain a data log for each
    key at the corresponding index.

    Also assumes all keys are unique, that data
    content is representable in floating-point format,
    and the base directory the CSV is stored in is
    descriptive of the experiment (used for env title).
    """

    def __init__(self, csv_path, xkey, xstart=0, xend=100, xitrv=1):
        """
        All other keys will be plotted vs the xkey.
        Can plot X slices of data (e.g. when log is large) using
        xstart, xend, and xitrv (x interval). xstart and xend
        should be 0-100 and are interpreted as percentages.
        """
        self.viz = Visdom()

        self.path = osp.abspath(csv_path)
        self.env_name = osp.split(osp.dirname(csv_path))[-1]

        self.xs = xstart
        self.xe = xend
        self.xi = xitrv

        self.ptr = 0

        keys = self._get_keys()
        assert xkey in keys
        self.xkey = xkey
        self.xidx = keys.index(xkey)

        self.ykeys = keys
        self.ykeys.remove(xkey)

        self.plotters = [
            LinePlotter(self.viz, xkey, ykey, ykey, env_name=self.env_name)
            for ykey in self.ykeys
        ]

    def _content(self, linestr):
        tokens = linestr.strip().split(",")
        return list(filter(lambda x: x != "", tokens))

    def _extract(self, data):
        """
        Extracts X/Y curves from readlines data (list of line strings).

        Returns corresponding lists of X and Y curves, where each pair
        has the same data length. This first filtered based on provided
        X points (omitted points are marked as "None") and subsequently
        filtered based on the provided points for each Y curve.
        """
        tokens = [self._content(d) for d in data]
        keep_mask = [[x != "None" for x in line] for line in tokens]
        float_data = [
            [float(x) if x != "None" else float("nan") for x in line] for line in tokens
        ]

        keep_mask = np.asarray(
            keep_mask
        ).T  # transpose of data len 'N' and curve 'K': (N, K) --> (K, N)
        curves = np.asarray(float_data).T  # similarly transposed

        def pullout_xvec(matrix, xidx):
            xvec = matrix[xidx, :]
            ymat = np.concatenate((matrix[:xidx, :], matrix[xidx + 1 :, :]), axis=0)
            return xvec, ymat

        xmask, ymasks = pullout_xvec(keep_mask, self.xidx)
        xcurve, ycurves = pullout_xvec(curves, self.xidx)

        xcurve = xcurve[xmask]
        ycurves = ycurves[:, xmask]  # subset of Y with existing X
        ymasks = ymasks[:, xmask]

        xcurves_plot_ready = []
        ycurves_plot_ready = []
        for kidx in range(len(ycurves)):
            ycurve, ymask = ycurves[kidx], ymasks[kidx]
            xcurves_plot_ready.append(xcurve[ymask])
            ycurves_plot_ready.append(ycurve[ymask])

        return xcurves_plot_ready, ycurves_plot_ready

    def _get_keys(self):
        with open(self.path, "r", newline="") as f:
            keys = self._content(f.readline())
            self.ptr = f.tell()
        return keys

    def _get_data(self):
        with open(self.path, "r", newline="") as f:
            f.seek(self.ptr)
            data = f.readlines()
            self.ptr = f.tell()

        if data:
            xcurves, ycurves = self._extract(data)
        else:  # data == [] when no new logs have occurred (i.e. in live mode)
            xcurves = ycurves = None

        return xcurves, ycurves

    def _get_x_slice(self, xcrv):
        size = len(xcrv)
        start_idx = round((float(self.xs) / 100) * size)
        end_idx = round((float(self.xe) / 100) * size)
        return slice(start_idx, end_idx, self.xi)

    def run_once(self, ypltkeys=None, use_x_slice=True):
        """
        Primary data gathering function.
        Reads all CSV lines since last pointer (stop)
        location (which on the first call is right before
        the first data line after the keys), slices the
        data if requested, and plots each curve in ypltkeys
        (or all available curves if nothing is specified).
        """
        xcurves, ycurves = self._get_data()

        if xcurves is not None:
            assert ycurves is not None

            for xcrv, ycrv, pltr in zip(xcurves, ycurves, self.plotters):
                assert len(xcrv) == len(ycrv)
                # proceed if plottable data was extracted for this curve/slice
                # and if ylabel includes at least one plot key substring
                if len(xcrv) > 0 and (
                    not ypltkeys or any([ypk in pltr.ylabel for ypk in ypltkeys])
                ):
                    # apply custom X slice, if specified
                    if use_x_slice:
                        xslc = self._get_x_slice(xcrv)
                        xcrv = xcrv[xslc]
                        ycrv = ycrv[xslc]

                    # ship to Visdom server
                    pltr.plot(xcrv, ycrv)

    def run_continuous(self, ypltkeys=None, period=1.0):
        """
        Repeatedly send data to Visdom server in batches
        as the CSV log is updated. X-slicing specification
        is disabled right now since data is received in
        batches of varying sizes, requiring non-trivial
        index manipulation.
        """
        while True:
            self.run_once(ypltkeys, use_x_slice=False)
            time.sleep(period)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("once", "live"),
        help="plot log file once or plot live data continuously",
    )
    parser.add_argument(
        "exp_path",
        help="path to experiment log, either a CSV or a directory containing 'progress.csv'",
    )
    parser.add_argument(
        "y_curves",
        nargs="*",
        help="which curves to plot (multi arg), accepts partial matches; if nothing is provided, all available data is plotted",
    )
    parser.add_argument(
        "-x", "--x_curve", default=DEFAULT_X_KEY, help="X-axis to plot curves over"
    )
    parser.add_argument(
        "-s",
        "--x_start_percent",
        type=int,
        default=0,
        help="plot curves starting after this % of X points ('once' mode only)",
    )
    parser.add_argument(
        "-e",
        "--x_stop_percent",
        type=int,
        default=100,
        help="stop curve plotting at this % of X points ('once' mode only)",
    )
    parser.add_argument(
        "-i",
        "--x_interval",
        type=int,
        default=1,
        help="plot points every 'x_interval' steps along X-axis starting at 'x_start_idx' ('once' mode only)",
    )
    args = parser.parse_args()

    if args.mode != "once" and (
        args.x_start_percent != 0 or args.x_stop_percent != 100 or args.x_interval != 1
    ):
        raise ValueError(
            "X-slicing ('x_start_percent', 'x_stop_percent', 'x_interval') currently is only supported for 'once' mode, and not 'live' mode"
        )

    csv_path = (
        args.exp_path
        if args.exp_path.endswith(".csv")
        else osp.join(args.exp_path, "progress.csv")
    )

    logpltr = LogPlotter(
        csv_path,
        args.x_curve,
        xstart=args.x_start_percent,
        xend=args.x_stop_percent,
        xitrv=args.x_interval,
    )

    if args.mode == "once":
        logpltr.run_once(args.y_curves)
    else:  # mode == "live"
        logpltr.run_continuous(args.y_curves)
