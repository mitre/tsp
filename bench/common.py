
from timeit import timeit


def bench_model_callable(call_gen, problems, repeats=1, max_enc=1, max_dec=1, max_crt=1, device=None):
    print(f"Averaging benchmarks over {repeats} repeats")
    stat_dict = {}
    
    for e in range(max_enc):
        f = call_gen(problems, e+1, 1, 1, device=device)
        tag, time = f"Enc-{e+1}", timeit(f, number=repeats) / repeats
        stat_dict[tag] = time
        print(tag, f" {time:.4f}")

    for d in range(max_dec):
        f = call_gen(problems, 1, d+1, 1, device=device)
        tag, time = f"Dec-{d+1}", timeit(f, number=repeats) / repeats
        stat_dict[tag] = time
        print(tag, f" {time:.4f}")

    for c in range(max_crt):
        f = call_gen(problems, 1, 1, c+1, device=device)
        tag, time = f"Crt-{c+1}", timeit(f, number=repeats) / repeats
        stat_dict[tag] = time
        print(tag, f" {time:.4f}")

    return stat_dict