#!/bin/bash

# IO params
scriptname="mc_ecc_onesample_scaling.py"
alpha="0.05"
M=2000
nsignature=1000
ntest=2000
 # create file with commands that will be run

for n in 1000
do
dim=2
for distid in {0..29}
do
         python ${scriptname} --n ${n} --dim ${dim} --n_signature ${nsignature} --n_test ${ntest} --M ${M} --output_dp ${outputdp} --alpha ${alpha} --distid ${distid} &
done
done
wait
