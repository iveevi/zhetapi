#ifndef ESSENTIALS_H_
#define ESSENTIALS_H_

#ifdef ZHP_CUDA // Cuda active

#define __cuda_dual__ __host__ __device__

#else

#define __cuda_dual__

#endif // Cuda active

// Use when we want to define a new variable cudaError_t error
#define __cuda_check_error()					\
	cudaError_t error = cudaGetLastError();			\
	if (error != cudaSuccess) {				\
		printf("CUDA error: %s\n",			\
				cudaGetErrorString(error));	\
		exit(-1);					\
	}

// Use when cudaError_t error has already been defined
#define __cuda_check_perror()					\
	error = cudaGetLastError();				\
	if (error != cudaSuccess) {				\
		printf("CUDA error: %s\n",			\
				cudaGetErrorString(error));	\
		exit(-1);					\
	}

#endif
