import os
import os.path as osp
import numpy as np
from pprint import pprint
import torch

import tsp


def get_log_dir(suffix=""):
    pack_path = osp.dirname(tsp.__file__)
    install_path = osp.dirname(pack_path)
    run_path = osp.join(install_path, "runs")
    return run_path


def get_latest_check_path(name, log_dir=None):
    """
    Finds path of latest checkpoint based on
    run name and logging directory.
    """
    if log_dir is not None:
        log_dir = osp.abspath(osp.expanduser(log_dir))
    else:
        log_dir = get_log_dir()

    run_path = osp.join(log_dir, name)
    file_names = [osp.basename(fp) for fp in os.listdir(run_path)]
    chkpts = filter(lambda x: x.startswith("params") and x.endswith(".pt"), file_names)

    if not chkpts:
        raise ValueError(
            f"Attempting to load checkpoint path from run with no checkpoints in expected format: '{run_path}'"
        )

    chkpts = sorted(chkpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    return osp.join(log_dir, name, chkpts[-1])


def get_resume_info(csv_path):
    """Returns last iteration and time of a csv log."""
    with open(csv_path, "r") as csv:
        logs = csv.readlines()
    num_logs = len(logs) - 1  # -1 from header line

    def extract_content(linestr):
        tokens = linestr.strip().split(",")
        return list(filter(lambda x: x != "", tokens))

    header_tokens = extract_content(logs[0])
    last_tokens = extract_content(logs[-1])

    req_keys = sorted(header_tokens)

    itr_idx = header_tokens.index("iteration")
    time_idx = header_tokens.index("time")

    last_itr = int(last_tokens[itr_idx])
    last_time = float(last_tokens[time_idx])

    return last_itr, last_time, req_keys, num_logs


class Logger:
    """
    A global logger class that can be called anywhere.
    Logger should be initialized before starting the runner.

    The keys logged before the first call to Logger.dump
    define the required keys for subsequent calls to this function.

    Output is comma seperated in a format used by tools such as
    viskit: https://github.com/vitchyr/viskit
    """

    store: dict
    req_keys: list
    path_dir: str
    csv_name: str
    csv_path: str
    chk_name: str
    chk_base_path: str
    last_chk_path: str
    prefix: str
    count: int
    setup: bool = False
    dummy: bool = False

    @classmethod
    def init(cls, rank, path_dir, csv_name="progress", chk_name="params", resume=False):
        """
        path_dir: string of directory to store logging info (creates if does not exist)
        csv_name: string of csv file name where logs will get dumped
        chk_name: string of pt file name where model checkpoints will get stored
        resume: boolean letting logger know to append to existing log (and not raise exception)
        """
        if cls.setup:
            raise Exception("Global Logger already initialized!")

        if not osp.exists(path_dir):
            os.makedirs(path_dir)

        cls.path_dir = path_dir
        cls.csv_name = csv_name
        cls.csv_path = osp.join(
            osp.abspath(osp.expanduser(path_dir)), csv_name + str(rank) + ".csv"
        )

        # If the file is short, just get rid of it
        if not resume and osp.exists(cls.csv_path):

            with open(cls.csv_path, "r") as f:
                n_lines = len(f.readlines())

            # If its less than 1000, probably okay to just delete
            if n_lines <= 1000:
                os.remove(cls.csv_path)

        try:
            open(cls.csv_path, "a" if resume else "x")
        except:
            raise Exception(
                f"Could not create log file at {cls.csv_path}\nPath either already exists or is illegal"
            )

        cls.chk_name = chk_name
        cls.chk_base_path = osp.join(
            osp.abspath(osp.expanduser(path_dir)), chk_name
        )  # finalized with itr info
        cls.last_chk_path = None  # always keep checkpoints when resuming

        cls.store = dict()
        cls.prefix = None
        cls.setup = True

        if resume:
            last_itr, last_time, req_keys, num_logs = get_resume_info(cls.csv_path)
            cls.req_keys = req_keys
            cls.count = num_logs
        else:
            last_itr, last_time = 0, 0
            cls.req_keys = None
            cls.count = 0

        return last_itr, last_time

    @classmethod
    def dummy_init(cls):
        """
        Allows calling Logger without actually logging.
        Useful for testing routines.
        """
        if cls.setup:
            raise Exception("Global Logger already initialized!")

        cls.store = dict()
        cls.req_keys = None
        cls.prefix = None
        cls.count = 0
        cls.setup = True
        cls.dummy = True

    @classmethod
    def log(cls, key, value):
        lkey = cls.prefix + "_" + key if cls.prefix else key
        if not cls.dummy:
            assert (
                lkey not in cls.store
            ), f"Overwriting log key '{lkey}' before calling Logger.dump()!"

        cls.store[lkey] = value

    @classmethod
    def log_stat(cls, key, values):
        """Logs mean, median, min, max, and std for numpy and torch tensors"""
        if values is not None:
            mean = values.mean().item()
            median = (
                np.median(values).item()
                if isinstance(values, np.ndarray)
                else values.median().item()
            )
            minv = values.min().item()
            maxv = values.max().item()
            std = values.std().item()
        else:
            mean = None
            median = None
            minv = None
            maxv = None
            std = None

        cls.log(key + "_avg", mean)
        cls.log(key + "_med", median)
        cls.log(key + "_min", minv)
        cls.log(key + "_max", maxv)
        cls.log(key + "_std", std)

    @classmethod
    def scope(cls, prefix):
        """Set scope of subsequent logging with prefix"""
        cls.prefix = str(prefix) if prefix else None

    @classmethod
    def dump(cls):
        if not cls.dummy:
            with open(cls.csv_path, "a") as csvl:
                sorted_keys = sorted(cls.store.keys())

                # first dump, fix required keys and log them
                if cls.count == 0:
                    cls.req_keys = sorted_keys
                    for key in cls.req_keys:
                        csvl.write(str(key) + ",")
                    csvl.write("\n")

                # check keys match required set
                if sorted_keys != cls.req_keys:
                    missing_keys = set(cls.req_keys).difference(set(sorted_keys))
                    extra_keys = set(sorted_keys).difference(cls.req_keys)
                    if missing_keys:
                        raise RuntimeError(
                            f"Missing values when logging: {missing_keys}"
                        )
                    else:
                        raise RuntimeError(
                            f"Unexpected values when logging: {extra_keys}"
                        )

                # log values
                for _, value in sorted(cls.store.items()):
                    csvl.write(str(value) + ",")
                csvl.write("\n")

        else:  # print to terminal when dummy logging
            pass
            # pprint(cls.store, indent=2)

        cls.count += 1
        cls.store = dict()

    @classmethod
    def checkpoint(cls, state_dict, iteration, keep_prev=False):
        """Save model state dict"""
        if not cls.dummy:
            curr_chk_path = cls.chk_base_path + f"_{iteration}.pt"
            torch.save(state_dict, curr_chk_path)

            if not keep_prev and cls.last_chk_path is not None:
                os.remove(cls.last_chk_path)

            cls.last_chk_path = curr_chk_path
