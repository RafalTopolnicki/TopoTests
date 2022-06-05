import numpy as np
import sys
import os
import argparse
from randomvariables import get_random_variables

# setting path
sys.path.append("../topotests/")
sys.path.append("../multiKS/")
from topotests import TopoTest_twosample
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

np.random.seed(1)


def data_row(args, true_label, alter_label, accpecth0, pvals, threshold):
    return {
        "n": args.n,
        "dim": args.dim,
        "norm": args.norm,
        "M": args.M,
        "permutations": args.permutations,
        "significance_level": args.alpha,
        "true_dist": true_label,
        "alter_dist": alter_label,
        "threshold": threshold,
        "accpect_h0": accpecth0,
        "pvals": pvals,
    }


def run_mc(rvs, args):
    outputfilename = f"twosample_dim={args.dim}_n={args.n}_norm={args.norm}.csv"
    outputfilepath = os.path.join(args.output_dp, outputfilename)
    results = []
    for rv_true in rvs:
        logging.info(f"Start true distribution: {rv_true.label} n={args.n} dim={args.dim} norm={args.norm}")
        for rv_alter in rvs:
            logging.info(f"Start distribution true: {rv_true.label} alter: {rv_alter.label}")
            tt_threshold = []
            tt_pval = []
            for mcloop in range(args.M):
                X1 = rv_true.rvs(args.n)
                X2 = rv_alter.rvs(args.n)
                tt = TopoTest_twosample(X1=X1, X2=X2, norm=args.norm, loops=args.permutations)
                tt_threshold.append(tt[0])
                tt_pval.append(tt[1])

            accepth0 = np.mean(np.array(tt_pval) > args.alpha)
            results.append(
                data_row(
                    args=args,
                    true_label=rv_true.label,
                    alter_label=rv_alter.label,
                    accpecth0=accepth0,
                    pvals=tt_pval,
                    threshold=tt_threshold,
                )
            )
            # write results to file after each test
            df = pd.DataFrame(results)
            df.to_csv(outputfilepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="data size i.e. number of data points in each sample",
    )
    parser.add_argument("--dim", type=int, required=True, help="dimension of the data points")
    parser.add_argument("--permutations", type=int, required=True, help="number of permutations")
    parser.add_argument("--M", type=int, required=True, help="number of MC repetitions")
    parser.add_argument("--output_dp", type=str, default="", help="where to dump output")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument(
        "--norm",
        type=str,
        default="sup",
        choices=["sup", "l1", "l2"],
        help="norm used to compute distance between ECCs",
    )
    args = parser.parse_args()

    rvs = get_random_variables(dim=args.dim)

    run_mc(rvs, args)


if __name__ == "__main__":
    main()
