import numpy as np
import sys
import os
import argparse
from randomvariables import get_random_variables

# setting path
sys.path.append("../topotests/")
from pdwdtest import  PDWDTest_onesample
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
    outputfilename = f"pdwdtest_dim={args.dim}_n={args.n}_persistence_dim={args.persistence_dim}.csv"
    outputfilepath = os.path.join(args.output_dp, outputfilename)
    results = []
    df_done = None
    if os.path.exists(outputfilepath):
        logging.info(f"OUTPUTFILE FOUND! Will continue from here.")
        df_done = pd.read_csv(outputfilepath).iloc[:, 1:]
        outputfilepath = os.path.join(args.output_dp, outputfilename+'.cont') #fixme: this works only for one restart

    for rv_true in rvs:
        logging.info(f"WD-1s: Start true distribution: {rv_true.label} n={args.n} dim={args.dim} persistence_dim={args.persistence_dim}")
        rv_true_done = True
        for rv_alter in rvs:
            if df_done is not None and not np.any(np.logical_and(df_done.true_dist == rv_true.label, df_done.alter_dist == rv_alter.label)):
                #at least one alternative distribution is missing - need to repeat whole simulation
                #for that true distribution
                rv_true_done = False
        if df_done is not None and rv_true_done:
            logging.info(f"WD-1s: All with true distribution: {rv_true.label} n={args.n} dim={args.dim} persistence_dim={args.persistence_dim} done!")
            continue

        pdwd_test = PDWDTest_onesample(
            n=args.n,
            dim=args.dim,
            significance_level=args.alpha,
            persistence_dim=args.persistence_dim
        )
        # train TopoTest
        pdwd_test.fit(rv=rv_true, n_signature=args.n_signature, n_test=args.n_test)
        for rv_alter in rvs:
            if df_done is not None and np.any(np.logical_and(df_done.true_dist == rv_true.label, df_done.alter_dist == rv_alter.label)):
                logging.info(f"WD-1s: Skip distribution true: {rv_true.label} alter: {rv_alter.label}")
                continue
            else:
                logging.info(f"WD-1s: Start distribution true: {rv_true.label} alter: {rv_alter.label}")
                # generate samples from alternative distributions
                samples = [rv_alter.rvs(args.n) for i in range(args.M)]
                # get list of H0 acceptances and p-values
                tt_out = pdwd_test.predict(samples)
                results.append(
                    data_row(
                        args=args,
                        true_label=rv_true.label,
                        alter_label=rv_alter.label,
                        accpecth0=tt_out[0],
                        pvals=tt_out[1],
                        threshold=pdwd_test.representation_threshold,
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
    parser.add_argument("--persistence_dim", type=int, default=1, help="dim of persistance dim")


    args = parser.parse_args()

    rvs = get_random_variables(dim=args.dim)

    run_mc(rvs, args)


if __name__ == "__main__":
    main()
