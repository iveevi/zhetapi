#ifndef NETWORK_CUH_
#define NETWORK_CUH_

// Enable CUDA member functions
#define ZHP_CUDA

// Engine headers
#include <network.hpp>

// CUDA headers
#include <cuda/activation.cuh>
#include <cuda/error.cuh>
#include <cuda/kernels.cuh>
#include <cuda/lock.cuh>
#include <cuda/matrix.cuh>
#include <cuda/optimizer.cuh>
#include <cuda/vector.cuh>

// Derive these from device properties later on
#define MAX_BLOCKS	65535L
#define MAX_THREADS	1024L

namespace zhetapi {

namespace ml {


// Set all elements of the (gradient) matrix to 0
template <class T>
__global__
void reset(T *J, size_t size)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid; i < size; i+= threads)
		J[i] = 0;
}

template <class T>
__global__
void scale_down(T *J, T c, size_t size)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid; i < size; i+= threads)
		J[i] /= c;
}


// Aplpy gradient
template <class T>
__global__
void apply_gradient_k(T *W, T *M, T *J, T alpha, T mu, size_t size)
{
	size_t threads = blockDim.x * gridDim.x;
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = tid; i < size; i+= threads) {
		M[i] = mu * M[i] - alpha * J[i];
		W[i] += M[i];
	}
}

template <class T>
void adjusted(
		T *d_A,
		T *d_W,
		T *d_M,
		size_t net_size,
		size_t *h_rows,
		size_t *h_cols,
		T mu)
{
	// TODO: Stuff this all into a single kernel
	size_t offset = 0;
	for (int i = 0; i < net_size - 1; i++) {
		size_t blocks = min(MAX_BLOCKS, h_rows[i]);
		size_t threads = min(MAX_THREADS, h_cols[i]);

		__mmc_fma <<<blocks, threads>>> ((d_A + offset), (d_W + offset),
				(d_M + offset), mu, h_rows[i] * h_cols[i]);

		offset += h_rows[i] * h_cols[i];
	}

	cudaDeviceSynchronize();
	cudaCheckError(nullptr);
}

template <class T>
void compute_isolated_parallelized(
		const Vector <T> &in,
		T *d_W,
		size_t *h_rows,
		size_t *h_cols,
		ml::Activation <T> **acts,
		T **aloc,
		T **atloc,		// Temporary array
		T **zloc,
		size_t *arows,
		size_t *zrows,
		size_t size)
{
	cuda_host_to_device_memcpy(atloc[0], in.arr(), sizeof(T) * (arows[0] - 1));

	size_t i = 0;
	size_t offset = 0;
	while (i < size - 1) {
		size_t blocks = min(MAX_BLOCKS, h_rows[i]);
		size_t threads = min(MAX_THREADS, h_cols[i]);

		cudaDeviceSynchronize();

		/* TODO: Instead of launching a kernel for this, do
		 * the extra work in the multiplication kernel (ie. remove the
		 * neccesity of atloc).
		 */
		__apt_one_cpy <<<1, 1>>> (aloc[i], atloc[i], arows[i]);

		cudaDeviceSynchronize();

		__vmv_mult <<<blocks, threads>>> (atloc[i + 1], (d_W + offset), aloc[i],
				h_rows[i], h_cols[i]);
		
		cudaDeviceSynchronize();
		
		// Apply activations
		__act_dual_cpy <<<1, 1>>> (atloc[i + 1], zloc[i], acts[i + 1],
				zrows[i]);
		
		cudaDeviceSynchronize();
		
		offset += h_rows[i] * h_cols[i];
		
		// Progress the loop
		i++;
	}

	// This is the output of the network
	aloc[i] = atloc[i];
}

template <class T, class F>
void gradient_and_accumulate_isolated_parallelized(
		T *d_W,
		size_t *h_rows,
		size_t *h_cols,
		size_t elems,
		Activation <T> **acts,
		T **aloc,
		T **atloc,
		T **zloc,
		T **dloc,
		size_t *arows,
		size_t *zrows,
		size_t *drows,
		size_t size,
		size_t osize,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		F cmp,
		T *d_J,
		double &gopt,
		size_t &gpass)
{
	// Compute the actual value
	compute_isolated_parallelized(
			in,
			d_W,
			h_rows,
			h_cols,
			acts,
			aloc,
			atloc,
			zloc,
			arows,
			zrows,
			size);

	// Create the output vector
	T *dout;

	cuda_device_alloc(&dout, sizeof(T) * osize);
	cuda_host_to_device_memcpy(dout, out.arr(), sizeof(T) * osize);

	// Allocate gopt and gpass copies
	double *dgopt;
	size_t *dgpass;

	cuda_device_alloc(&dgopt, sizeof(double));
	cuda_host_to_device_memcpy(dgopt, &gopt, sizeof(double));
	
	cuda_device_alloc(&dgpass, sizeof(size_t));
	cuda_host_to_device_memcpy(dgpass, &gpass, sizeof(size_t));
	
	// Run stats kernel (move resources to unified memory for this)
	__acc_opt <<<1, 1>>> (dloc[0], aloc[size - 1], dout, osize, cmp, opt, dgopt,
			dgpass);

	cudaDeviceSynchronize();

	// Construction the Jacobian using backpropogation
	size_t offset = elems - (h_rows[size - 2] * h_cols[size - 2]);

	size_t blocks = 1;
	size_t threads = 1;
	for (int i = size - 2; i >= 0; i--) {
		size_t ai = size - (i + 2);

		if (i < size - 2) {
			threads = min(MAX_THREADS, h_cols[i]);

			__rmt_mtv_mult <<<1, threads>>> (dloc[ai], d_W + offset,
					dloc[ai - 1], h_rows[i + 1], h_cols[i + 1]);
			
			offset -= h_rows[i] * h_cols[i];
		}

		threads = min(MAX_THREADS, zrows[i]);
		__st_vv_shur <<<blocks, threads>>>  (dloc[ai], zloc[i],
				zrows[i]);
		
		cudaDeviceSynchronize();
		
		blocks = min(MAX_BLOCKS, h_rows[i]);
		threads = min(MAX_THREADS, h_cols[i]);

		__st_mvvt_add <<<blocks, threads>>> ((d_J + offset), dloc[ai], aloc[i],
				zrows[i], arows[i]);

		cudaDeviceSynchronize();
	}

	// Copy stats back
	cuda_device_to_host_memcpy(&gopt, dgopt, sizeof(double));
	cuda_device_to_host_memcpy(&gpass, dgpass, sizeof(size_t));
}

// Training a batch with CUDA
template <class T>
template <class F>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::cuda_batch(
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		Activation <T> **acts,
		Optimizer <T> *opt,
		T *d_W,
		T *d_M,
		T *d_A,
		T *d_J,
		size_t *d_rows,
		size_t *d_cols,
		size_t *h_rows,		// Remove this
		size_t *h_cols,		// and this
		size_t elems,
		T **aloc,
		T **atloc,
		T **zloc,
		T **dloc,
		size_t *arows,
		size_t *zrows,
		size_t *drows,
		T alpha,
		T mu,
		F cmp,
		bool printing)
{
	using namespace std;

	// Setup timers
	std::chrono::high_resolution_clock::time_point fstart;
	std::chrono::high_resolution_clock::time_point lstart;
	std::chrono::high_resolution_clock::time_point lend;
	std::chrono::high_resolution_clock::time_point fend;
	
	std::chrono::high_resolution_clock clk;

	// Start full timer
	fstart = clk.now();
	
	// Reinterpret constants
	size_t data_size = ins.size();
	size_t net_size = __size;

	// Allocate the gradients
	double gopt = 0;
	size_t gpass = 0;

	// Kernel launch variables
	size_t blocks;
	size_t threads;

	// TODO: Set appropriate blocks and threads
	blocks = min(MAX_BLOCKS, max(1L, elems/MAX_THREADS));
	threads = min(MAX_THREADS, elems);

	reset <<<blocks, threads>>> (d_J, elems);

	cudaDeviceSynchronize();

	lstart = clk.now();

	zhetapi::ml::adjusted(
				d_A,
				d_W,
				d_M,
				net_size,
				h_rows,
				h_cols,
				0.7);

	for (int i = 0; i < data_size; i++) {
		Vector <T> in = ins[i];
		Vector <T> out = outs[i];

		gradient_and_accumulate_isolated_parallelized(
					d_A,
					h_rows,
					h_cols,
					elems,
					acts,
					aloc,
					atloc,
					zloc,
					dloc,
					arows,
					zrows,
					drows,
					net_size,
					__osize,
					in,
					out,
					opt,
					cmp,
					d_J,
					gopt,
					gpass);
	}

	// Stop kernel time
	lend = clk.now();

	// TODO: change name and blocks/threads 
	scale_down <<<blocks, threads>>> (d_J, T(data_size), elems);

	cudaDeviceSynchronize();

#ifdef ZHP_GRAD_DEBUG

	cout << "d_J:" << endl;
	
	__print_array <<<1, 1>>> (d_J, elems);

	cudaDeviceSynchronize();

#else

	// TODO: Change name and blocks/threads
	apply_gradient_k <<<blocks, threads>>> (d_W, d_M, d_J, alpha, 0.7, elems);

#endif
	
	// Stop full timer
	fend = clk.now();
		
	double ltime = std::chrono::duration_cast
		<std::chrono::microseconds> (lend - lstart).count();
	double ftime = std::chrono::duration_cast
		<std::chrono::microseconds> (fend - fstart).count();

	return {gpass, gopt, ltime, ftime};
}

// Epoch training with CUDA
template <class T>
template <class F>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::cuda_epochs(
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t iterations,
		size_t batch_size,
		T alpha,
		F cmp,
		bool printing)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

	/* Allocate memory to the GPU early:
	 * allocate the arrays only to the GPU,
	 * so that the addresses of the arrays are
	 * accessible from the CPU.
	 */
	T **aloc;
	T **atloc;
	T **zloc;
	T **dloc;

	size_t *arows;
	size_t *zrows;
	size_t *drows;

	// alloc_gpu(aloc, atloc, zloc, dloc, arows, zrows, drows);

	aloc = new T *[__size];
	atloc = new T *[__size];
	zloc = new T *[__size - 1];
	dloc = new T *[__size - 1];
	
	arows = new size_t[__size];
	zrows = new size_t[__size - 1];
	drows = new size_t[__size - 1];
	
	size_t i;

	// As
	i = 0;
	while (i < __size - 1) {
		arows[i] = __layers[i].first + 1;

		cudaMalloc(&aloc[i], sizeof(T) * arows[i]);
		cudaMalloc(&atloc[i++], sizeof(T) * (arows[i] - 1));
	}

	arows[i] = __osize;
	
	cudaMalloc(&aloc[i], sizeof(T) * __osize);
	cudaMalloc(&atloc[i], sizeof(T) * __osize);

	// Zs
	i = 0;
	while (i < __size - 2) {
		zrows[i] = __layers[i + 1].first;

		cudaMalloc(&zloc[i++], sizeof(T) * zrows[i]);
	}
	
	zrows[i] = __osize;

	cudaMalloc(&zloc[i], sizeof(T) * __osize);


	// Deltas
	i = 0;

	drows[i] = __osize;

	cudaMalloc(&dloc[i++], sizeof(T) * drows[i]);
	while (i < __size - 1) {
		drows[i] = zrows[__size - (i + 2)];

		cudaMalloc(&dloc[i++], sizeof(T) * drows[i]);
	}

	// Activations
	Activation <T> **acts = new Activation <T> *[__size];
	for (int i = 0; i < __size; i++) {
		cuda_device_alloc(&acts[i], sizeof(Activation <T>));
		cuda_host_to_device_memcpy(acts[i], __layers[i].second,
				sizeof(Activation <T>));
	}
	
	// Optimizer
	Optimizer <T> *opt;

	cuda_device_alloc(&opt, sizeof(Optimizer <T>));
	cuda_host_to_device_memcpy(opt, __cost, sizeof(Optimizer <T>));
	
	// Weights and momentum
	size_t *d_rows;
	size_t *d_cols;
	
	size_t *h_rows;
	size_t *h_cols;
	
	T *d_W;
	T *d_M;
	T *d_A;
	T *d_J; // Gradient array
	
	T *h_W;
	T *h_M;
	
	// Allocation constants
	size_t bytes;
	size_t elems;
	
	// Allocate dimensions to GPU
	h_rows = new size_t[__size - 1];
	h_cols = new size_t[__size - 1];

	bytes = sizeof(size_t) * (__size - 1);

	cudaMalloc(&d_rows, bytes);
	cudaMalloc(&d_cols, bytes);

	bytes = 0;
	elems = 0;
	for (size_t i = 0; i < __size - 1; i++) {
		h_rows[i] = __weights[i].get_rows();
		h_cols[i] = __weights[i].get_cols();

		elems += h_rows[i] * h_cols[i];
	}

	cudaMemcpy(d_rows, h_rows, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cols, h_cols, bytes, cudaMemcpyHostToDevice);

	// Allocate matrices to GPU as flattened arrays
	h_W = new T[elems];
	h_M = new T[elems];
	
	bytes = sizeof(T) * elems;

	cudaMalloc(&d_W, bytes);
	cudaMalloc(&d_M, bytes);
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_J, bytes);

	size_t k = 0;
	for (size_t i = 0; i < __size - 1; i++) {
		for (size_t jx = 0; jx < h_rows[i]; jx++) {
			for (size_t jy = 0; jy < h_cols[i]; jy++) {
				h_W[k] = __weights[i][jx][jy];
				h_M[k++] = __momentum[i][jx][jy];
			}
		}
	}
	
	cudaMemcpy(d_W, h_W, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);

	// Split the batch
	std::vector <DataSet <T>> ins_batched = split(ins, batch_size);
	std::vector <DataSet <T>> outs_batched = split(outs, batch_size);

	T lr = alpha;

	T t_err = 0;
	double t_ktime = 0;
	double t_ftime = 0;
	size_t t_passed = 0;
	
	size_t total = 0;
	size_t passed;
	double kt;
	double ft;
	T err;
	for (int i = 0; i < iterations; i++) {
		if (printing) {
			std::cout << std::string(20, '-')
				<< std::endl
				<< "\nEpoch #" << (i + 1)
				<< " (" << lr
				<< ")" << std::endl;
		}
			
		passed = 0;
		err = 0;
		kt = 0;
		ft = 0;

		for (int j = 0; j < ins_batched.size(); j++) {
			TrainingStatistics result =
				cuda_batch <F> (
						ins_batched[j],
						outs_batched[j],
						acts,
						opt,
						d_W,
						d_M,
						d_A,
						d_J,
						d_rows,
						d_cols,
						h_rows,
						h_cols,
						elems,
						aloc,
						atloc,
						zloc,
						dloc,
						arows,
						zrows,
						drows,
						lr,
						0.7,
						cmp,
						printing);

			// Save results
			passed += result.__passed;
			err += result.__cost;
			kt += result.__kernel_time;
			ft += result.__full_time;

			lr = alpha * pow(0.1, (++total)/50000.0);
		}

		t_passed += passed;
		t_err += err;
		t_ktime += kt;
		t_ftime += ft;
		
		if (printing) {
			std::cout << "\nTotal cost:\t"
				<< err << std::endl
				<< "Kernel time:\t" << kt/1000
				<< " ms" << std::endl
				<< "Full time:\t" << ft/1000
				<< " ms" << std::endl
				<< "Cases passed:\t"
				<< passed
				<< "/" << ins.size() << " ("
				<< 100 * ((double) passed)/ins.size()
				<< "%)" << std::endl;
		}
	}

	// TODO: Copy the device weight arrays back to the network

	// Deallocate aloc and zloc
	for (int i = 0; i < __size; i++) {
		cuda_device_free(aloc[i]);

		// Skip the last element (duplicate pointer)
		if (i < __size - 1)
			cuda_device_free(atloc[i]);
	}
	
	for (int i = 0; i < __size - 1; i++) {
		cuda_device_free(dloc[i]);
		cuda_device_free(zloc[i]);
	}

	delete[] aloc;
	delete[] atloc;
	delete[] zloc;

	delete[] arows;
	delete[] zrows;
	delete[] drows;

	// Deallocate activations
	for (size_t i = 0; i < __size; i++)
		cuda_device_free(acts[i]);
	
	delete[] acts;

	// Deallocate optimizer
	cuda_device_free(opt);

	// Deallocate weights and momentum
	cudaFree(d_rows);
	cudaFree(d_cols);

	// Copy network state back
	cuda_device_to_host_memcpy(h_W, d_W, bytes);
	cuda_device_to_host_memcpy(h_M, d_M, bytes);
	
	k = 0;
	for (size_t i = 0; i < __size - 1; i++) {
		for (size_t jx = 0; jx < h_rows[i]; jx++) {
			for (size_t jy = 0; jy < h_cols[i]; jy++) {
				__weights[i][jx][jy] = h_W[k];
				__momentum[i][jx][jy] = h_M[k++];
			}
		}
	}

	delete[] h_W;
	delete[] h_M;
	
	cudaFree(d_W);
	cudaFree(d_M);
	cudaFree(d_A);
	cudaFree(d_J);

	delete[] h_rows;
	delete[] h_cols;

	return {t_passed, t_err, t_ktime, t_ftime};
}

}

}

#endif
