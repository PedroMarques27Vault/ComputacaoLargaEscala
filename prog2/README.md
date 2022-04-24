## Matrix Determinant Calculation with Multithreading

### How to compile:

	gcc -Wall -g -O3 -o prog2 main.c matrixutils.c sharedregion.c -pthread -lm

### How to run:

Arguments:

	-h --- print usage
	-f --- filename to process
	-n --- number of threads
	-k --- size of fifo queue in monitor

Example:

	./prog2 -f shortMatrix/mat128_64.bin -f shortMatrix/mat128_32.bin -k 8 -n 4
