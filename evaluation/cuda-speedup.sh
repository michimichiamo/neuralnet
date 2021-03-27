#!/bin/bash

# cuda-speedup.sh - Evaluating speedup
#
# Run with:
# ./cuda-speedup.sh N K
#
# NB: N should be set as to comply with the amount of available GPU memory
#     For the lab machine, a possible configuration is given by N=1000 K=150

N=$1 # 1st layer inputs
K=$2 # Number of layers

N1=(1 10 100) # 1st multiplier
N2=(1 2 5) # 2nd multiplier

CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

echo -e "N0=$N"
echo -e "pu\tN\tt1\tt2\tt3\tt4\tt5"
echo -e "cpu"
for n1 in ${N1[@]}; do
	for n2 in ${N2[@]}; do
		echo -e "\t$((n1*n2))x\t\c"
		for rep in `seq 5`; do
			OMP_EXEC_TIME="$( OMP_NUM_THREADS=$CORES ./nn_omp $((n1*n2*N)) $K | sed 's/Execution time //' )"
		    echo -e "${OMP_EXEC_TIME}\t\c"
		done
		echo ""
	done
done
echo -e "gpu"
for n1 in ${N1[@]}; do
	for n2 in ${N2[@]}; do
		echo -e "\t$((n1*n2))x\t\c"
		for rep in `seq 5`; do
			CUDA_EXEC_TIME="$(./nn_cuda $((n1*n2*N)) $K | sed 's/Execution time //' )"
		    echo -e "${CUDA_EXEC_TIME}\t\c"
		done
		echo ""
	done
done