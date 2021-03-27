#!/bin/bash

# omp.sh - Evaluating speedup and strong scaling efficiency
#
# Run with:
# ./omp.sh N K

echo -e "p\tt1\tt2\tt3\tt4\tt5"

N=$1 # 1st layer inputs
K=$2 # Number of layers
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

for p in `seq $CORES`; do
    echo -e "$p\t\c"
    for rep in `seq 5`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p ./nn_omp $N $K | sed 's/Execution time //' )"
        echo -e "${EXEC_TIME}\t\c"
    done
    echo ""
done
