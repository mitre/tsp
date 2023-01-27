
import json
import os.path as osp

import tsp
from tsp.utils import get_coords
from tsp.model.gpt_style import TspGptAcModel
from tsp.select import sample_select
from tsp.utils import batch_dist_gather

from common import bench_model_callable


device = 1
prob_size = 50
batch_size = 32
repeats = 10
max_lyr = 1


def gpt_callable_gen(problems, n_enc=1, n_dec=1, n_crt=1, device=None):
    """n_enc and n_crt are just placeholders for common API"""
    model = TspGptAcModel(
        dim_model=128,
        num_dec_layers=n_dec
    )

    if device is not None:
        model.to(device)
        problems = problems.to(device)

    _, select_idxs, log_probs, values = model(problems, sample_select)
    sel_log_probs = batch_dist_gather(log_probs, select_idxs)

    def func():
        (sel_log_probs.mean() + values.mean()).backward(retain_graph=True)

    return func


if __name__ == "__main__":
    problems = get_coords(batch_size, prob_size)

    # test gpu mem is sufficient first with single run at max
    print(f"Testing max layers: {max_lyr}")
    gpt_callable_gen(problems, n_dec=1, device=device)()

    # benchmark sweeping one layer type at a time
    times = bench_model_callable(
        gpt_callable_gen,
        problems, 
        repeats, 
        1, 
        max_lyr,
        1,
        device
    )

    out_name = osp.join(osp.dirname(osp.dirname(tsp.__file__)), 
        "bench", f"gpt_backward_{prob_size}n_{batch_size}b_{repeats}r_{max_lyr}D.json")

    with open(out_name, "w") as fh:
        json.dump(times, fh, indent=4)
