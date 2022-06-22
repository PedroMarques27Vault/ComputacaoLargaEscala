CU_APPS=matrixDeterminantRows

all: ${CU_APPS}

%: %.cu
	nvcc -O2 -Wno-deprecated-gpu-targets -o $@ $< matrix_utils_row.cu
clean:
	rm -f ${CU_APPS}

