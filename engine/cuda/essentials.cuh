#ifndef ESSENTIALS_H_
#define ESSENTIALS_H_

#ifdef ZHP_CUDA

#define __cuda_dual_prefix __host__ __device__

#else

#define __cuda_dual_prefix

#endif

#endif
