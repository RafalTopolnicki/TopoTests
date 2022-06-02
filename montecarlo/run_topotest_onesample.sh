#!/bin/bash
# this requires script requires parallel: https://www.gnu.org/software/parallel/

# how many jobs to run at the same time
jobs=24

# IO params
scriptname="mc_topotest_onesample.py"
outputdp="../results/"
commandsfile="commands.txt"

# set parameters that are the same for all runs
method="approximate"
complex_type="alpha"
alpha="0.05"
M=1000
nsignature=1000
ntest=1000
# create file with commands that will be run

if [ -f ${commandsfile} ]; then
  rm "${commandsfile}"
fi

for dim in 1 2 3 5
do
  for n in 100 250 500 1000 2500 5000
  do
    for norm in "sup" "l1"
    do
      echo "python ${scriptname} --n ${n} --dim ${dim} --n_signature ${nsignature} --n_test ${ntest} --M ${M} --output_dp ${outputdp} --norm ${norm} --alpha ${alpha} --method ${method}" --complex_type ${complex_type} >> ${commandsfile}
    done # end norm-loop
  done # end n-loop
done # end dim-loop

parallel --jobs ${jobs} < ${commandsfile}