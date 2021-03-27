#!/bin/bash

# omp-weak.sh - Evaluating weak scaling efficiency
#
# Run with:
# ./omp-weak.sh N

echo -e "K\tp\tt1\tt2\tt3\tt4\tt5"

N=$1 # 1st layer inputs
K=(10 50 100 150) # Number of layers
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores


for k in ${K[@]}; do
	echo -e "$k"
	for p in `seq $CORES`; do
    	echo -e "\t$p\t\c"
    	for rep in `seq 5`; do
    	    EXEC_TIME="$( OMP_NUM_THREADS=$p ./nn_omp $((p*N)) $k | sed 's/Execution time //' )"
    	    echo -e "${EXEC_TIME}\t\c"
    	done
    	echo ""
	done
done