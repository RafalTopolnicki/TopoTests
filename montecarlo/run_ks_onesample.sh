#!/bin/bash
# this requires script requires parallel: https://www.gnu.org/software/parallel/

# how many jobs to run at the same time
jobs=14

# IO params
scriptname="mc_ks_onesample.py"
outputdp="../results/"
commandsfile="commands_ks_1sample.txt"

# set parameters that are the same for all runs
M=1000

# create file with commands that will be run

if [ -f ${commandsfile} ]; then
  rm "${commandsfile}"
fi

for dim in 1 2 3
do
  for n in 100 250 500 1000 2500
  do
    echo "python ${scriptname} --n ${n} --dim ${dim} --M ${M} --output_dp ${outputdp}" >> ${commandsfile}
  done # end n-loop
done # end dim-loop

parallel --jobs ${jobs} < ${commandsfile}