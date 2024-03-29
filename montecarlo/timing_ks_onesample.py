import numpy as np
import sys
import os
import argparse
from randomvariables import get_random_variables
from scipy import stats

# setting path
sys.path.append("../multiKS/")
import pandas as pd
import logging
from multiKS import multiKS
import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

np.random.seed(1)


def data_row(args, true_label, alter_label, dks):
    return {
        "n": args.n,
        "dim": args.dim,
        "M": args.M,
        "true_dist": true_label,
        "alter_dist": alter_label,
        "dks": dks,
    }


def run_ks(rvs, args):
    outputfilename = f"timing_ks_dim={args.dim}_n={args.n}.csv"
    outputfilepath = os.path.join(args.output_dp, outputfilename)
    results = []

    for rv_true in [rvs[0]]:
        time_start = time.perf_counter()
        logging.info(f"KS-1S: Start true distribution: {rv_true.label} n={args.n} dim={args.dim}")
        # train TopoTest
        for rv_alter in rvs:
            logging.info(f"KS-1S: Start distribution true: {rv_true.label} alter: {rv_alter.label}")
            # generate samples from alternative distributions
            dkss = []
            # this can be run in parallel - see e.g. MC_AllDistr3D notebook
            for loop in range(args.M):
                sample = rv_alter.rvs(args.n)
                dks = multiKS(sample, rv_true.cdf)
                dkss.append(dks)
            # get list of H0 acceptances and p-values
            results.append(
                data_row(
                    args=args,
                    true_label=rv_true.label,
                    alter_label=rv_alter.label,
                    dks=dkss
                )
            )
            # write results to file after each test
            df_data = pd.DataFrame(results)
            df_data.to_csv(outputfilepath)
        logging.info(f"KS-1s: DONE distribution: {rv_true.label} n={args.n} dim={args.dim}")
        time_done = time.perf_counter()
        logging.info(f"KS-1s: {args}")
        logging.info(f"KS-1s: start: {time_start} done: {time_done} len: {time_done-time_start}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="data size i.e. number of data points in each sample",
    )
    parser.add_argument("--dim", type=str, required=True, help="dimension of the data points")
    parser.add_argument("--M", type=int, required=True, help="number of MC repetitions")
    parser.add_argument("--output_dp", type=str, default="", help="where to dump output")
    args = parser.parse_args()

    rvs = get_random_variables(dim=args.dim)

    run_ks(rvs, args)


if __name__ == "__main__":
    main()
