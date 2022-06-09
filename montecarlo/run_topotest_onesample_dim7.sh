#!/bin/bash
# this requires script requires parallel: https://www.gnu.org/software/parallel/

# how many jobs to run at the same time
jobs=4

# IO params
scriptname="mc_topotest_onesample.py"
outputdp="../results/"
commandsfile="commands_1s_dim7.txt"

# set parameters that are the same for all runs
method="approximate"
alpha="0.05"
M=100
nsignature=100
ntest=100
# create file with commands that will be run

if [ -f ${commandsfile} ]; then
  rm "${commandsfile}"
fi

for dim in 7
do
  for n in 100 250 500 1000
  do
    for norm in "sup"
    do
      echo "python ${scriptname} --n ${n} --dim ${dim} --n_signature ${nsignature} --n_test ${ntest} --M ${M} --output_dp ${outputdp} --norm ${norm} --alpha ${alpha} --method ${method}" >> ${commandsfile}
    done # end norm-loop
  done # end n-loop
done # end dim-loop

parallel --jobs ${jobs} < ${commandsfile}