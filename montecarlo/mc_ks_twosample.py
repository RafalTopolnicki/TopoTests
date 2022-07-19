import numpy as np
import sys
import os
import argparse
from randomvariables import get_random_variables

# setting path
sys.path.append("../multiKS/")
sys.path.append("../2DKS_fast/")
import pandas as pd
import logging
from multiKS import multiKS2s
from KS2Dfast import ks2d2s

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

np.random.seed(1)

def data_row(args, true_label, alter_label, dks, pval):
    return {
        "n": args.n,
        "dim": args.dim,
        "M": args.M,
        "true_dist": true_label,
        "alter_dist": alter_label,
        "dks": dks,
        "pval": pval
    }


def run_mc(rvs, args):
    outputfilename = f"ks_2sample_dim={args.dim}_n={args.n}.csv"
    outputfilepath = os.path.join(args.output_dp, outputfilename)
    # check if outputfile exists
    df = None
    if os.path.exists(outputfilepath):
        logging.info(f"OUTPUTFILE FOUND! Will continue from here.")
        df = pd.read_csv(outputfilepath).iloc[:, 1:]
        outputfilepath = os.path.join(args.output_dp, outputfilename+'.cont') #fixme: this works only for one restart
    results = []
    rvs_n = len(rvs)
    for rv_true_id in range(rvs_n-1):
        rv_true = rvs[rv_true_id]
        logging.info(f"KS-2S: Start true distribution: {rv_true.label} n={args.n} dim={args.dim}")
        # train TopoTest
        for rv_alter_id in range(rv_true_id, rvs_n):
            rv_alter = rvs[rv_alter_id]
            if df is not None:
                if np.any(np.logical_and(df.true_dist == rv_true.label, df.alter_dist == rv_alter.label)):
                    logging.info(f"KS-2S: Skipping {rv_true.label} alter: {rv_alter.label}")
                    continue
            logging.info(f"KS-2S: Start distribution true: {rv_true.label} alter: {rv_alter.label}")
            # generate samples from alternative distributions
            dkss = []
            pvals = []
            # this can be run in parallel - see e.g. MC_AllDistr3D notebook
            for loop in range(args.M):
                sample0 = rv_true.rvs(args.n)
                sample1 = rv_alter.rvs(args.n)
                if args.dim == 1:
                    dks, pval = stats.ks_2samp(sample0[:, 0], sample1[:, 0])
                elif args.dim == 2:
                    dks, pval = ks2d2s(sample0, sample1)
                else:
                    dks, pval = multiKS2s(sample0, sample1)
                dkss.append(dks)
                pvals.append(pval)
            # get list of H0 acceptances and p-values
            results.append(
                data_row(
                    args=args,
                    true_label=rv_true.label,
                    alter_label=rv_alter.label,
                    dks=dkss,
                    pval=pvals
                )
            )
            # write results in oposite order for easier processing later
            results.append(
                data_row(
                    args=args,
                    true_label=rv_alter.label,
                    alter_label=rv_true.label,
                    dks=dkss,
                    pval=pvals
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
    parser.add_argument("--M", type=int, required=True, help="number of MC repetitions")
    parser.add_argument("--output_dp", type=str, default="", help="where to dump output")
    args = parser.parse_args()

    rvs = get_random_variables(dim=args.dim)

    run_mc(rvs, args)


if __name__ == "__main__":
    main()
