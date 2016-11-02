#The -ccbin argument is needed for the setup on IT-South 015

mandel_cuda: mandel_cuda.cu
	nvcc -ccbin=g++-4.8 -O3 mandel_cuda.cu -o mandel_cuda

clean:
	rm mandel_cuda
