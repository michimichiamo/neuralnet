#!/bin/sh

# Questo script exegue il programma omp-matmul sfruttando OpenMP con
# un numero di core variabile da 1 a 8 (estremi inclusi); ogni
# esecuzione considera sempre la stessa dimensione dell'input, quindi
# i tempi misurati possono essere usati per calcolare speedup e strong
# scaling efficiency. Ogni esecuzione viene ripetuta 5 volte; vengono
# stampati a video i tempi di esecuzione di tutte le esecuzioni.

# NB: La dimensione del problema (PROB_SIZE = 800, nel nostro caso il
# numero di righe o colonne della matrice) e' scelta per ottenere dei
# tempi di esecuzione "umani" sulla macchina disi-hpc

echo "omp\tcuda"
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores


OMP_EXEC_TIME="$( OMP_NUM_THREADS=$CORES ./nn_omp | sed 's/Execution time //' )"
echo "${OMP_EXEC_TIME}\t\c"
CUDA_EXEC_TIME="$(./nn_cuda | sed 's/Execution time //' )"
echo "${CUDA_EXEC_TIME}\t\c"
echo ""

echo "$(./test)"