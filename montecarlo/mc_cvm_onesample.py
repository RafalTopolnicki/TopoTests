import numpy as np
import sys
import os
import argparse

import scipy.stats

from randomvariables import get_random_variables

# setting path
sys.path.append("../multiKS/")
import pandas as pd
import logging
from multiKS import multiKS

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

np.random.seed(1)


def data_row(args, true_label, alter_label, cvm_pvalue):
    return {
        "n": args.n,
        "dim": args.dim,
        "M": args.M,
        "true_dist": true_label,
        "alter_dist": alter_label,
        "cvm_pvalue": cvm_pvalue,
    }


def run_mc(rvs, args):
    outputfilename = f"cvm_dim={args.dim}_n={args.n}.csv"
    outputfilepath = os.path.join(args.output_dp, outputfilename)
    # check if outputfile exists
    df = None
    if os.path.exists(outputfilepath):
        logging.info(f"OUTPUTFILE FOUND! Will continue from here.")
        df = pd.read_csv(outputfilepath).iloc[:, 1:]
        outputfilepath = os.path.join(args.output_dp, outputfilename+'.cont') #fixme: this works only for one restart
    results = []
    for rv_true in rvs:
        logging.info(f"CVM-1S: Start true distribution: {rv_true.label} n={args.n} dim={args.dim}")
        # train TopoTest
        for rv_alter in rvs:
            if df is not None and np.any(np.logical_and(df.true_dist == rv_true.label, df.alter_dist == rv_alter.label)):
               logging.info(f"CVM-1S: Skipping {rv_true.label} alter: {rv_alter.label}")
            else:
                logging.info(f"CVM-1S: Start distribution true: {rv_true.label} alter: {rv_alter.label}")
                # generate samples from alternative distributions
                dcvms = []
                # this can be run in parallel - see e.g. MC_AllDistr3D notebook
                for loop in range(args.M):
                    sample = rv_alter.rvs(args.n)
                    dcvm = scipy.stats.cramervonmises(sample[:, 0], rv_true.cdf)
                    dcvms.append(dcvm.pvalue)
                # get list of H0 acceptances and p-values
                results.append(
                    data_row(
                        args=args,
                        true_label=rv_true.label,
                        alter_label=rv_alter.label,
                        cvm_pvalue=dcvms
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
    parser.add_argument("--M", type=int, required=True, help="number of MC repetitions")
    parser.add_argument("--output_dp", type=str, default="", help="where to dump output")
    args = parser.parse_args()

    #if args.dim != 1:
    #    raise NotImplementedError(f'--dim must be 1')

    rvs = get_random_variables(dim=args.dim)
    run_mc(rvs, args)


if __name__ == "__main__":
    main()
