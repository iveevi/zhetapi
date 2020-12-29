#ifndef KERNELS_H_
#define KERNELS_H_

namespace zhetapi {

namespace ml {

template <class T>
class Activation;

template <class T>
class Optimizer;

}

// For debugging only
template <class T>
__global__
void __print_array(T *arr, size_t size)
{
	printf("{");
	for (size_t i = 0; i < size; i++)
		printf("%f, ", arr[i]);
	printf("\b \b\b}\n");
}

template <class T>
__global__
void __mmc_fma(T *R, T *W, T *M, T c, size_t size)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid; i < size; i += threads)
		R[i] = W[i] + c * M[i];
}

template <class T>
__global__
void __vmv_mult(T *R, T *W, T *A, size_t rows, size_t cols)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid; i < rows; i += threads) {
		// Cache the row
		T *row = &(W[i * cols]);

		T acc = 0;

		for (size_t k = 0; k < cols; k++)
			acc += row[k] * A[k];

		R[i] = acc;
	}
}

// Append one (copy) kernel
template <class T>
__global__
void __apt_one_cpy(T *R, T *A, size_t size)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid == 0)
		R[0] = 1;

	for (size_t i = tid; i < size - 1; i += threads)
		R[i + 1] = A[i];
}

// Apply the activation (R) and its derivative (Rext)
// (Both copies are in place, no allocation done)
template <class T>
__global__
void __act_dual_cpy(T *R, T *Rext, ml::Activation <T> *act, size_t size)
{
	ml::Activation <T> *a = copy(act);
	ml::Activation <T> *ad = a->derivative();

	// Create vectors for each array (without duplicating resources)
	Vector <T> t(size, R);
	Vector <T> ts(size, Rext);

	ts.stable_transfer((*ad)(t));

	__syncthreads();
	
	Vector <T> ta = (*a)(t);

	t.stable_transfer((*a)(t));

	delete a;
	delete ad;
}

// Accumulate and apply optimizer kernel
template <class T, class F>
__global__
void __acc_opt(
		T *delta,
		T *actual,
		T *out,
		size_t size,
		F cmp,
		ml::Optimizer <T> *opt,
		double *gopt,
		size_t *gpass)
{
	Vector <T> v_act(size, actual);
	Vector <T> v_out(size, out);
	Vector <T> v_delta(size, delta);

	ml::Optimizer <T> *dopt = copy(opt);
	
	if (cmp(v_act, v_out))
		(*gpass)++;

	*gopt += (*dopt)(v_act, v_out)[0];

	ml::Optimizer <T> *ddopt = dopt->derivative();

	v_delta.stable_transfer((*ddopt)(v_out, v_act));

	delete ddopt;
	delete dopt;
}

// Stable vector-vector shur kernel
template <class T>
__global__
void __st_vv_shur(T *R, T *A, size_t size)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid; i < size; i += threads)
		R[i] *= A[i];
}

// Remove top, transposed matrix times vector kernel
template <class T>
__global__
void __rmt_mtv_mult(T *R, T *W, T *A, size_t rows, size_t cols)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid + 1; i < cols; i += threads) {
		T acc = 0;

		for (size_t k = 0; k < rows; k++) 
			acc += W[k * cols + i] * A[k];

		R[i - 1] = acc;
	}
}

// Multipy a vector and a transposed vector and add to a matrix
template <class T>
__global__
void __st_mvvt_add(T *R, T *V, T *Vt, size_t rows, size_t cols)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid; i < rows * cols; i += threads)
		R[i] += V[i / cols] * Vt[i % cols];
}

}

#endif
