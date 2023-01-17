"""
Tools interfacing external solvers (e.g. pyconcorde) 
with tsp, if installed.
"""

import torch
from contextlib import contextmanager
import ctypes
import os
import io
import sys
import tempfile

from tsp.utils import get_costs


class ConcordeSolver:
    """
    Solve 2D TSP problems using Concorde, which is
    currently the fastest exact TSP solver.

    bridge.ConcordeSolver wraps pyconcorde's TSPSolver which
    wraps the original Concorde C binaries.

    Internally scales problem domain to 1 million
    times its original scale to minimize error
    accumlation from edge-distance-rounding which
    occurs inside the Concorde binaries. Apparently
    this makes the TSP problem NP-complete rather
    than just NP-hard. See this GitHub issue:
    https://github.com/jvkersch/pyconcorde/issues/29
    """

    internal_scale = 1e6

    def __init__(self):
        try:
            from concorde.tsp import TSPSolver
        except:
            raise ImportError(
                "Unable to find pyconcorde; has this submodule been installed?"
            )

        self.SolverFactory = TSPSolver

    def __call__(self, problems):
        """
        Accepts (N, S, 2) shaped torch.Tensor problem stacks.
        """
        solution_tours = []
        solution_idxs = []
        solution_costs = []

        for prob in problems:
            xs = prob[:, 0].clone().numpy()
            ys = prob[:, 1].clone().numpy()

            prob_scale = max(xs.max(), ys.max()) - min(xs.min(), ys.min())
            scale_factor = self.internal_scale / prob_scale
            xs *= scale_factor
            ys *= scale_factor

            with open(os.devnull, "w+b") as nullout:
                with c_stdout_redirector(nullout), c_stderr_redirector(nullout):
                    solver = self.SolverFactory.from_data(xs, ys, "EUC_2D")
                    solution = solver.solve()

            sol_idxs = torch.tensor(solution.tour, dtype=torch.long)
            sol_tour = prob[sol_idxs]

            solution_idxs.append(sol_idxs)
            solution_tours.append(sol_tour)

        solution_idxs = torch.stack(solution_idxs, dim=0)
        solution_tours = torch.stack(solution_tours, dim=0)
        solution_costs = get_costs(solution_tours)

        return solution_tours, solution_idxs, solution_costs


# def redirect_c_stream(c_stream, p_stream, original_fd, redirect_fd):
#     """Redirect C stream to the given file descriptor."""
#     libc = ctypes.CDLL(None)
#     # Flush the C-level buffer
#     libc.fflush(c_stream)
#     # Flush and close Python stream - also closes the file descriptor (fd)
#     p_stream.close()
#     # Make original_fd point to the same file as redirect_fd
#     os.dup2(redirect_fd, original_fd)
#     # Update p_stream to point to the redirected fd
#     sys.stdout = io.TextIOWrapper(os.fdopen(original_fd, "wb"))


# def c_stream_redirector(orig_c_stream, orig_p_stream, redir_p_stream):
#     # The original fd. Usually 1 on POSIX systems.
#     original_fd = orig_p_stream.fileno()

#     # Save a copy of the original fd in saved_fd
#     saved_fd = os.dup(original_fd)
#     try:
#         # Create a temporary file and redirect stream to it
#         tfile = tempfile.TemporaryFile(mode="w+b")
#         redirect_c_stream(orig_c_stream, orig_p_stream, original_fd, tfile.fileno())
#         # Yield to caller, then redirect stream back to the saved fd
#         yield
#         redirect_c_stream(orig_c_stream, orig_p_stream, original_fd, saved_fd)
#         # Copy contents of temporary file to the given stream
#         tfile.flush()
#         tfile.seek(0, io.SEEK_SET)
#         redir_p_stream.write(tfile.read())
#     finally:
#         tfile.close()
#         os.close(saved_fd)


# @contextmanager
# def c_stdout_redirector(redir_p_stream):
#     libc = ctypes.CDLL(None)
#     c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
#     p_stdout = sys.stdout
#     yield from c_stream_redirector(c_stdout, p_stdout, redir_p_stream)


# @contextmanager
# def c_stderr_redirector(redir_p_stream):
#     libc = ctypes.CDLL(None)
#     c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")
#     p_stderr = sys.stderr
#     yield from c_stream_redirector(c_stderr, p_stderr, redir_p_stream)


@contextmanager
def c_stdout_redirector(stream):
    """
    Redirects stdout streams originating from C bindings.
    Taken from https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    """
    libc = ctypes.CDLL(None)
    c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")

    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, "wb"))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode="w+b")
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


@contextmanager
def c_stderr_redirector(stream):
    """
    Identical to c_stdout_redirector but for
    C stderr instead of C stdout.
    """
    libc = ctypes.CDLL(None)
    c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")

    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        libc.fflush(c_stderr)
        sys.stderr.close()
        os.dup2(to_fd, original_stderr_fd)
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, "wb"))

    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        tfile = tempfile.TemporaryFile(mode="w+b")
        _redirect_stderr(tfile.fileno())
        yield
        _redirect_stderr(saved_stderr_fd)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stderr_fd)
