import numpy as np
import sys
import os
import argparse
from randomvariables import get_random_variables

# setting path
sys.path.append("../topotests/")
sys.path.append("../multiKS/")
from topotests import TopoTest_onesample
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

np.random.seed(1)

def data_row(args, true_label, alter_label, accpecth0, pvals, threshold):
    return {
        "n": args.n,
        "dim": args.dim,
        "norm": args.norm,
        "complex_type": args.complex_type,
        "n_signature": args.n_signature,
        "n_test": args.n_test,
        "M": args.M,
        "method": args.method,
        "significance_level": args.alpha,
        "true_dist": true_label,
        "alter_dist": alter_label,
        "threshold": threshold,
        "accpect_h0": accpecth0,
        "pvals": pvals,
    }


def run_mc(rvs, args):
    outputfilename = f"dim={args.dim}_n={args.n}_norm={args.norm}_method={args.method}_complex={args.complex_type}.csv"
    outputfilepath = os.path.join(args.output_dp, outputfilename)
    results = []
    for rv_true in rvs:
        logging.info(f"Start true distribution: {rv_true.label} n={args.n} dim={args.dim} norm={args.norm}")
        topo_test = TopoTest_onesample(
            n=args.n,
            dim=args.dim,
            method=args.method,
            norm=args.norm,
            significance_level=args.alpha,
            complex_type=args.complex_type
        )
        # train TopoTest
        topo_test.fit(rv=rv_true, n_signature=args.n_signature, n_test=args.n_test)
        for rv_alter in rvs:
            logging.info(f"Start distribution true: {rv_true.label} alter: {rv_alter.label}")
            # generate samples from alternative distributions
            samples = [rv_alter.rvs(args.n) for i in range(args.M)]
            # get list of H0 acceptances and p-values
            tt_out = topo_test.predict(samples)
            results.append(
                data_row(
                    args=args,
                    true_label=rv_true.label,
                    alter_label=rv_alter.label,
                    accpecth0=tt_out[0],
                    pvals=tt_out[1],
                    threshold=topo_test.representation_threshold,
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
    parser.add_argument(
        "--method",
        type=str,
        default="approximate",
        choices=["approximate", "exact"],
        help="method of treating the ECC average",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="sup",
        choices=["sup", "l1", "l2"],
        help="norm used to compute distance between ECCs",
    )
    parser.add_argument(
        "--complex_type",
        type=str,
        default="alpha",
        choices=["alpha", "rips"],
        help="type of complex used to build the simplex tree",
    )

    args = parser.parse_args()

    rvs = get_random_variables(dim=args.dim)

    run_mc(rvs, args)


if __name__ == "__main__":
    main()
