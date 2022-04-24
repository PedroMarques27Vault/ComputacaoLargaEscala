## Text Processing with Multithreading

### How to compile:

	gcc -Wall -g -O3 -o prog1 main.c sharedRegion.c textProcUtils.c

### How to run:

Arguments:

	-h --- print usage
	-f --- filename to process
	-n --- number of threads
	-m --- maximum number of bytes per chunk

Example:

	./prog1 -f texts/text0.txt -f texts/text1.txt -n 8 -m 2000
