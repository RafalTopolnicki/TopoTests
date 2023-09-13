import numpy as np
import sys
import os
import argparse
from randomvariables import get_random_variables

# setting path
sys.path.append("../topotests/")
#from topotests import TopoTest_onesample
from topotests.topotests import TopoTest_onesample
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

np.random.seed(1)


def data_row(args, true_label, alter_label, accpecth0, pvals, threshold):
    return {
        "n": args.n,
        "dim": args.dim,
        "n_signature": args.n_signature,
        "n_test": args.n_test,
        "M": args.M,
        "significance_level": args.alpha,
        "true_dist": true_label,
        "alter_dist": alter_label,
        "threshold": threshold,
        "accpect_h0": accpecth0,
        "pvals": pvals,
    }


def run_mc(rvs, args):
    rv_true = rvs[args.distid]
    null_distr_label = rv_true.label

    logging.info(f"ECC-1s: Start true distribution: {rv_true.label} n={args.n} dim={args.dim}")

    topotest = TopoTest_onesample(
        n=args.n,
        dim=args.dim,
        significance_level=args.alpha,

    )
    # train TopoTest
    topotest.fit(rv=rv_true, n_signature=args.n_signature, n_test=args.n_test)
    n_tests = [int(0.1*args.n), int(0.25*args.n), int(0.5*args.n), int(args.n), int(2*args.n), int(5*args.n), int(10*args.n)]
    for n_test in n_tests:
        results = []
        for rv_alter in rvs:
            outputfilename = f"ecctest_dim={args.dim}_n={args.n}_null={null_distr_label}_ntest={n_test}.csv"
            outputfilepath = os.path.join(args.output_dp, outputfilename)

            logging.info(f"ECC-1s: Start distribution true: {rv_true.label} alter: {rv_alter.label}")
            # generate samples from alternative distributions
            samples = [rv_alter.rvs(n_test) for i in range(args.M)]
            # get list of H0 acceptances and p-values
            tt_out = topotest.predict(samples)
            results.append(
                data_row(
                    args=args,
                    true_label=rv_true.label,
                    alter_label=rv_alter.label,
                    accpecth0=tt_out[0],
                    pvals=tt_out[1],
                    threshold=topotest.representation_threshold,
                )
            )
            # write results to file after each test
            df_data = pd.DataFrame(results)
            df_data.to_csv(outputfilepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="data size i.e. number of data points in each sample",
    )
    parser.add_argument("--dim", type=str, required=True, help="dimension of the data points")
    parser.add_argument(
        "--n_signature",
        type=int,
        required=True,
        help="number of samples used to compute mean ECC",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        required=True,
        help="number of samples used to compute threshold and estimate pvalue",
    )
    parser.add_argument("--M", type=int, required=True, help="number of MC repetitions")
    parser.add_argument("--output_dp", type=str, default="", help="where to dump output")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument("--distid", type=int, required=True, help="which distribution consider as null")

    args = parser.parse_args()

    rvs = get_random_variables(dim=args.dim)

    if args.distid < 0 or args.distid > len(rvs):
        raise ValueError("Parameter distid is wrong!")

    run_mc(rvs, args)


if __name__ == "__main__":
    main()
