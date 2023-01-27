
import json
import os.path as osp

import tsp
from tsp.utils import get_coords
from tsp.model.monty_style import TspMsAcModel
from tsp.select import sample_select

from common import bench_model_callable


device = 1
prob_size = 50
batch_size = 32
repeats = 10
max_enc = 1
max_dec = 1
max_crt = 1


def monty_callable_gen(problems, n_enc, n_dec, n_crt, device=None):
    model = TspMsAcModel(
        dim_model=128,
        num_enc_layers=n_enc,
        num_dec_layers=n_dec,
        num_crt_layers=n_crt,
    )

    if device is not None:
        model.to(device)
        problems = problems.to(device)

    def func():
        model(problems, sample_select)

    return func


if __name__ == "__main__":
    problems = get_coords(batch_size, prob_size)

    # test gpu mem is sufficient first with single run at each max
    print(f"Testing max enc layers: {max_enc}")
    monty_callable_gen(problems, max_enc, 1, 1, device)()
    
    print(f"Testing max dec layers: {max_dec}")
    monty_callable_gen(problems, 1, max_dec, 1, device)()

    print(f"Testing max crt layers: {max_crt}")
    monty_callable_gen(problems, 1, 1, max_crt, device)()

    # benchmark sweeping one layer type at a time
    times = bench_model_callable(
        monty_callable_gen,
        problems, 
        repeats, 
        max_enc, 
        max_dec,
        max_crt,
        device
    )

    out_name = osp.join(osp.dirname(osp.dirname(tsp.__file__)), 
        "bench", f"monty_forward_{prob_size}n_{batch_size}b_{repeats}r_{max_enc}E_{max_dec}D_{max_crt}C.json")

    with open(out_name, "w") as fh:
        json.dump(times, fh, indent=4)
