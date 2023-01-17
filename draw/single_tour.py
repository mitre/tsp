import argparse
from matplotlib import pyplot as plt
import torch

from tsp.utils import get_coords
from tsp.agent import TspAgent, TspRandomAgent, TspOracleAgent
from tsp.model.monty_style import TspMontyStyleModel

from base import plot_tsp


def get_tour(problem, solver, model_path=None):
    if solver == "random":
        rand_agent = TspRandomAgent()
        _, select_idxs = rand_agent.solve(problem)

    elif solver == "exact":
        # intractable for large problem sizes if pyconcorde not installed!
        orac_agent = TspOracleAgent()
        _, select_idxs = orac_agent.solve(problem)

    elif solver == "agent":
        model = TspMontyStyleModel()
        agent = TspAgent(model)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=agent.device)
            agent.load_state_dict(state_dict)

        agent.eval_mode()

        _, select_idxs = agent.solve(problem)

    else:
        raise ValueError(f"Unrecognized tour solver algorithm '{solver}'")

    return select_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "solver", choices=("random", "exact", "agent"), help="solver to use"
    )
    parser.add_argument("problem_size", type=int, help="number of nodes in TSP tour")
    parser.add_argument(
        "-m",
        "--model_path",
        default=None,
        help="path to agent model params, if applicable",
    )
    args = parser.parse_args()

    if args.solver != "agent" and args.model_path is not None:
        print(
            f"WARNING: solver '{args.solver}' does not take model path, ignoring input: {args.model_path}"
        )

    problem = get_coords(1, args.problem_size)
    select_idxs = get_tour(problem, args.solver, args.model_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tsp(problem.squeeze(), select_idxs.squeeze(), ax)
    plt.show()
