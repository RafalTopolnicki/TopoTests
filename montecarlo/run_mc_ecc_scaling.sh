#!/bin/bash

# IO params
scriptname="mc_ecc_onesample_scaling.py"
outputdp="../results.ecc_scaling/"

# set parameters that are the same for all runs
alpha="0.05"
M=1000
nsignature=1000
ntest=1000
# create file with commands that will be run

n=100
dim=2
for distid in 0 1 9
do
        python ${scriptname} --n ${n} --dim ${dim} --n_signature ${nsignature} --n_test ${ntest} --M ${M} --output_dp ${outputdp} --alpha ${alpha} --distid ${distid} &
done
wait