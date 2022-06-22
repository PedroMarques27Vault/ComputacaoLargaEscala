CU_APPS=matrixDeterminantCols

all: ${CU_APPS}

%: %.cu
	nvcc -O2 -Wno-deprecated-gpu-targets -o $@ $< matrix_utils_col.cu
clean:
	rm -f ${CU_APPS}
