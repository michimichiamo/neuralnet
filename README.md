# neuralnet
A simple, parallel, efficient neural network evaluator.

## Source files
Source files are ```nn_omp.c``` and ```nn_cuda.cu```.

## CPU implementation
Compile with: ```gcc -fopenmp nn_omp.c -o nn_omp```
Please note that some GCC versions might require the additional ```-lm``` flag to link the math library.
Run with: ```./nn_omp [N K]```
Values default to N=500000, K=150

## GPU implementation
Compile with: ```nvcc nn_cuda.cu -o nn_cuda```
Run with: ```./nn_cuda [N K]```
Values default to N=500000, K=150

## Evaluation
Bash scripts and ipython notebook used to assess the performance for both programs are provided inside the [evaluation](https://github.com/michimichiamo/neuralnet/evaluation) folder

## Report
A brief report describing the work is provided inside the [report](https://github.com/michimichiamo/neuralnet/report) folder. It includes the source LaTeX files as well as the [pdf](https://github.com/michimichiamo/neuralnet/report/report.pdf), along with the plots used within the document.
