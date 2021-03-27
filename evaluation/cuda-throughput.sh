#!/bin/bash

# cuda-throughput.sh - Evaluating throughput
#
# Run with:
# ./cuda-throughput.sh N K
#
# NB: N should be set as to comply with the amount of available GPU memory
#     For the lab machine, a possible configuration is given by N=1000 K=150

N=$1 # 1st layer inputs
K=$2 # Number of layers

N1=(1 10 100) # 1st multiplier
N2=(1 2 5) # 2nd multiplier

echo -e "$N"
echo -e "N\tt1\tt2\tt3\tt4\tt5"
for n1 in ${N1[@]}; do
	for n2 in ${N2[@]}; do
		echo -e "$((n1*n2))x\t\c"
		for rep in `seq 5`; do
		    EXEC_TIME="$(./nn_cuda $((n1*n2*N)) $K | sed 's/Execution time //' )"
		    echo -e "${EXEC_TIME}\t\c"
		done
		echo ""
	done
done