## Matrix Determinant Calculation with Multithreading

### Main Objective 
Given a file with matrices, calculate the determinant of each one, efficiently, by splitting processing load for worker threads.
### Multithreaded Implementation:
- Main thread processes the command line arguments.
- Main thread creates the worker threads.
- Main thread reads each matrix of each file.
- Each matrix is inserted in a Shared Memory in a FIFO.
- Workers retrieve and process the matrix, calculating the determinant.
- Workers insert results in the Shared Memory.
- When all files have been processed, the main thread retrieves and presents the results.

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
