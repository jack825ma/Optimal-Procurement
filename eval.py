from core import run_eval
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str,
                    help="The type of procedure corresponding to the name of config")
parser.add_argument("--n_run", type=int, default=3,
                    help="The number of runs of the same experiment with different random seeds")
parser.add_argument("--n_gpu", type=int, default=0,
                    help="The number of GPUs")
args = parser.parse_args()

for seed in range(args.n_run):
    run_eval.main(args.setting, seed)
