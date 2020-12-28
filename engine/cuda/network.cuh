#ifndef NETWORK_CUH_
#define NETWORK_CUH_

// Enable CUDA member functions
#define ZHP_CUDA

// Engine headers
#include <network.hpp>

// CUDA headers
#include <cuda/activation.cuh>
#include <cuda/error.cuh>
#include <cuda/lock.cuh>
#include <cuda/matrix.cuh>
#include <cuda/optimizer.cuh>
#include <cuda/vector.cuh>

namespace zhetapi {

namespace ml {

int opt = 0;

template <class T>
Matrix <T> *adjusted1(
		Matrix <T> *weights,
		Matrix <T> *momentum,
		size_t size,
		T mu)
{
	Matrix <T> *theta = new Matrix <T> [size - 1];
	for (int i = 0; i < size - 1; i++)
		theta[i] = weights[i];

	for (int i = 0; i < size - 1; i++)
		theta[i] += mu * momentum[i];

	return theta;
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
Matrix <T> *adjusted2(
		T *d_W,
		T *d_M,
		size_t net_size,
		size_t *h_rows,
		size_t *h_cols,
		T mu)
{
	Matrix <T> *theta = new Matrix <T> [net_size - 1];
	T **d_As = new T *[net_size - 1];

	size_t offset = 0;
	for (int i = 0; i < net_size - 1; i++) {
		T *d_A;

		cudaMalloc(&d_A, sizeof(T) * h_rows[i] * h_cols[i]);

		d_As[i] = d_A;

		size_t blocks = min(128L, h_rows[i]);
		size_t threads = min(128L, h_cols[i]);

		__mmc_fma <<<blocks, threads>>> (d_A, (d_W + offset), (d_M + offset), mu, h_rows[i] * h_cols[i]);

		offset += h_rows[i] * h_cols[i];
	}

	cudaDeviceSynchronize();
	for (int i = 0; i < net_size - 1; i++) {
		T *h_A = new T[h_rows[i] * h_cols[i]];

		cudaMemcpy(h_A, d_As[i], sizeof(T) * h_rows[i] * h_cols[i],
				cudaMemcpyDeviceToHost);

		theta[i] = Matrix <T> (h_rows[i], h_cols[i], h_A, true);

		cudaFree(d_As[i]);
	}

	return theta;
}

template <class T>
Vector <T> compute_isolated_parallelized1(
		const Vector <T> &in,
		Matrix <T> *weights,
		ml::Activation <T> **acts,
		Vector <T> *a,
		Vector <T> *z,
		size_t size)
{
	using namespace std;
	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size - 1) {
		a[i] = tmp.append_above(T (1));
		cout << "a[i] = " << a[i] << endl;

		prv = weights[i] * a[i];

		ml::Activation <T> *dev_act = copy(acts[i + 1]);

		tmp = (*dev_act)(prv);

		ml::Activation <T> *dev_dact = dev_act->derivative();

		z[i++] = (*dev_dact)(prv);
		cout << "z[i] = " << z[i - 1] << endl;

		delete dev_act;
		delete dev_dact;
	}

	a[i] = tmp;
	cout << "a[i] = " << a[i] << endl;
	
	return tmp;
}

template <class T>
__global__
void show(T *arr, size_t size)
{
	printf("arr = {");
	for (size_t i = 0; i < size; i++)
		printf("%f, ", arr[i]);
	printf("\b \b\b}\n");
}

template <class T>
__device__
void dev_show(T *arr, size_t size)
{
	printf("arr = {");
	for (size_t i = 0; i < size; i++)
		printf("%f, ", arr[i]);
	printf("\b \b\b}\n");
}

template <class T>
void host_show(T *arr, size_t size)
{
	printf("arr = {");
	for (size_t i = 0; i < size; i++)
		printf("%f, ", arr[i]);
	printf("\b \b\b}\n");
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

template <class T>
Vector <T> compute_isolated_parallelized2(
		const Vector <T> &in,
		Matrix <T> *weights,
		T *d_W,
		size_t *h_rows,
		size_t *h_cols,
		ml::Activation <T> **acts,
		Vector <T> *a,
		Vector <T> *z,
		T **aloc,
		T **atloc,
		T **zloc,
		size_t *arows,
		size_t *zrows,
		size_t size)
{
	using namespace std;
	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	size_t offset = 0;
	while (i < size - 1) {
		size_t blocks = min(128L, h_rows[i]);
		size_t threads = min(128L, h_cols[i]);

		a[i] = tmp.append_above(T (1));

		cout << "a[i] = " << a[i] << endl;

		//--------------------------------
		cudaMemcpy(aloc[i], a[i].arr(), sizeof(T) * arows[i],
				cudaMemcpyHostToDevice);

		show <<<1, 1>>> (aloc[i], arows[i]);

		cudaDeviceSynchronize();
		//--------------------------------

		prv = weights[i] * a[i];

		cout << "weights[i] = " << weights[i] << endl;
		cout << "prv = " << prv << endl;

		//--------------------------------
		T *p_arr;

		cudaMalloc(&p_arr, sizeof(T) * h_cols[i]);

		__vmv_mult <<<blocks, threads>>> (p_arr, (d_W + offset), aloc[i],
				h_rows[i], h_cols[i]);
		
		cudaDeviceSynchronize();
		
		show <<<1, 1>>> (p_arr, h_rows[i]);

		cudaDeviceSynchronize();
		//--------------------------------

		ml::Activation <T> *dev_act = copy(acts[i + 1]);

		tmp = (*dev_act)(prv);

		ml::Activation <T> *dev_dact = dev_act->derivative();

		z[i] = (*dev_dact)(prv);
		cout << "z[i] = " << z[i] << endl;

		delete dev_act;
		delete dev_dact;

		cout << "=======================================" << endl;

		offset += h_rows[i] * h_cols[i];

		i++;
	}

	a[i] = tmp;
	cout << "a[i] = " << a[i] << endl;
	
	return tmp;
}

template <class T, class F>
void gradient_and_accumulate_isolated_parallelized1(
		Matrix <T> *weights,
		Activation <T> **acts,
		size_t size,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		F cmp,
		Matrix <T> *grad,
		double &gopt,
		size_t &gpass)
{
	using namespace std;

	// Allocate memory for a and z
	Vector <T> *a = new Vector <T> [size];
	Vector <T> *z = new Vector <T> [size - 1];
	
	// Compute the actual value
	Vector <T> actual = compute_isolated_parallelized1(
			in,
			weights,
			acts,
			a,
			z,
			size);

	if (cmp(actual, out))
		gpass++;
	
	Optimizer <T> *dev_opt = copy(opt);

	gopt += (*dev_opt)(actual, out)[0];
	
	// Get the derivative of the cost
	Optimizer <T> *dev_dopt = dev_opt->derivative();
	
	// Construction the Jacobian using backpropogation
	Vector <T> delta = (*dev_dopt)(out, actual);
	for (int i = size - 2; i >= 0; i--) {
		if (i < size - 2) {
			delta = weights[i + 1].transpose() * delta;
			delta = delta.remove_top();
		}
		
		delta.stable_shur(z[i]);
		grad[i] += delta * a[i].transpose();
	}

	// Free resources
	delete[] a;
	delete[] z;

	delete dev_opt;
	delete dev_dopt;
}

template <class T, class F>
void gradient_and_accumulate_isolated_parallelized2(
		Matrix <T> *weights,
		T *d_W,
		size_t *h_rows,
		size_t *h_cols,
		Activation <T> **acts,
		T **aloc,
		T **zloc,
		size_t *arows,
		size_t *zrows,
		size_t size,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		F cmp,
		Matrix <T> *grad,
		double &gopt,
		size_t &gpass)
{
	using namespace std;

	// Allocate memory for a and z
	Vector <T> *a = new Vector <T> [size];
	Vector <T> *z = new Vector <T> [size - 1];
	
	// Compute the actual value
	Vector <T> actual = compute_isolated_parallelized2(
			in,
			weights,
			d_W,
			h_rows,
			h_cols,
			acts,
			a,
			z,
			aloc,
			zloc,
			arows,
			zrows,
			size);

	if (cmp(actual, out))
		gpass++;
	
	Optimizer <T> *dev_opt = copy(opt);

	gopt += (*dev_opt)(actual, out)[0];
	
	// Get the derivative of the cost
	Optimizer <T> *dev_dopt = dev_opt->derivative();
	
	// Construction the Jacobian using backpropogation
	Vector <T> delta = (*dev_dopt)(out, actual);
	for (int i = size - 2; i >= 0; i--) {
		if (i < size - 2) {
			delta = weights[i + 1].transpose() * delta;
			delta = delta.remove_top();
		}
		
		delta.stable_shur(z[i]);
		grad[i] += delta * a[i].transpose();
	}

	// Free resources
	delete[] a;
	delete[] z;

	delete dev_opt;
	delete dev_dopt;
}

// Training a batch with CUDA
template <class T>
template <class F, size_t blocks, size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::cuda_batch(
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		T **aloc,
		T **zloc,
		size_t *arows,
		size_t *zrows,
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
	Matrix <T> *J = new Matrix <T> [net_size - 1];
	for (int i = 0; i < net_size - 1; i++) {
		J[i] = Matrix <T> (__weights[i].get_rows(),
				__weights[i].get_cols(), T(0));
	}
	
	Activation <T> **acts = new Activation <T> *[__size];
	for (int i = 0; i < __size; i++)
		acts[i] = __layers[i].second;

	double gopt = 0;
	size_t gpass = 0;

	// GPU adjustments
	Matrix <T> *adj_weights;
	
	size_t *d_rows;
	size_t *d_cols;
	
	size_t *h_rows;
	size_t *h_cols;
	
	T *d_W;
	T *d_M;
	T *d_A;
	
	T *h_W;
	T *h_M;
	T *h_A;
	
	if (opt) {
		// Allocation constants
		size_t bytes;
		size_t elems;
		
		// Allocate dimensions to GPU
		h_rows = new size_t[net_size - 1];
		h_cols = new size_t[net_size - 1];

		bytes = sizeof(size_t) * (net_size - 1);

		cudaMalloc(&d_rows, bytes);
		cudaMalloc(&d_cols, bytes);

		bytes = 0;
		elems = 0;
		for (size_t i = 0; i < net_size - 1; i++) {
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

		size_t k = 0;
		for (size_t i = 0; i < net_size - 1; i++) {
			for (size_t jx = 0; jx < h_rows[i]; jx++) {
				for (size_t jy = 0; jy < h_cols[i]; jy++) {
					h_W[k] = __weights[i][jx][jy];
					h_M[k++] = __momentum[i][jx][jy];
				}
			}
		}
		
		cudaMemcpy(d_W, h_W, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
		
		lstart = clk.now();
		
		adj_weights = zhetapi::ml::adjusted2(
					d_W,
					d_M,
					net_size,
					h_rows,
					h_cols,
					0.7);
		
		// Copy the adjusted weights to the GPU
		h_A = new T[elems];
		
		cudaMalloc(&d_A, bytes);

		k = 0;
		for (size_t i = 0; i < net_size - 1; i++) {
			for (size_t jx = 0; jx < h_rows[i]; jx++) {
				for (size_t jy = 0; jy < h_cols[i]; jy++)
					h_A[k++] = adj_weights[i][jx][jy];
			}
		}

		cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	} else {
		// Start kernel timer
		lstart = clk.now();

		adj_weights = zhetapi::ml::adjusted1(
					__weights,
					__momentum,
					net_size,
					0.7);
	}

	for (int i = 0; i < data_size; i++) {
		Vector <T> in = ins[i];
		Vector <T> out = outs[i];

		if (opt) {
			gradient_and_accumulate_isolated_parallelized2(
						adj_weights,
						d_A,
						h_rows,
						h_cols,
						acts,
						aloc,
						zloc,
						arows,
						zrows,
						net_size,
						in,
						out,
						__cost,
						cmp,
						J,
						gopt,
						gpass);
		} else {
			gradient_and_accumulate_isolated_parallelized1(
						adj_weights,
						acts,
						net_size,
						in,
						out,
						__cost,
						cmp,
						J,
						gopt,
						gpass);
		}
	}

	// Stop kernel time
	lend = clk.now();

	for (int i = 0; i < __size - 1; i++)
		J[i] /= data_size;
	
	if (printing) {
		using namespace std;
		cout << "Javg:" << endl;
		for (int i = 0; i < __size - 1; i++)
			cout << "\t" << J[i] << endl;
	} else {
		apply_gradient(J, alpha, 0.7);
	}

	delete[] adj_weights;
	delete[] acts;
	delete[] J;

	if (opt) {
		delete[] h_rows;
		delete[] h_cols;
		
		delete[] h_W;
		delete[] h_M;

		cudaFree(d_rows);
		cudaFree(d_cols);
		
		cudaFree(d_W);
		cudaFree(d_M);
	}
	
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
template <class F, size_t blocks, size_t threads>
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

	size_t *arows;
	size_t *zrows;
	
	// TODO: Remove opt everywhere once done testing
	if (opt) {
		aloc = new T *[__size];
		atloc = new T *[__size];
		zloc = new T *[__size - 1];
		
		arows = new size_t[__size];
		zrows = new size_t[__size - 1];

		using namespace std;
		cout << "__size = " << __size << endl;
		
		size_t i;

		i = 0;
		while (i < __size - 1) {
			arows[i] = __layers[i].first + 1;

			cudaMalloc(&aloc[i++], sizeof(T) * (__layers[i].first + 1));
			cudaMalloc(&atloc[i++], sizeof(T) * (__layers[i].first + 1));
		}

		arows[i] = __osize;
		
		cudaMalloc(&aloc[i], sizeof(T) * __osize);
		cudaMalloc(&atloc[i], sizeof(T) * __osize);

		i = 0;
		while (i < __size - 2) {
			zrows[i] = __layers[i].first;

			cudaMalloc(&zloc[i++], sizeof(T) * (__layers[i].first));
		}

		zrows[i] = __osize;

		cudaMalloc(&zloc[i], sizeof(T) * __osize);

		using namespace std;
		cout << "a:" << endl;
		for (int i = 0; i < __size; i++)
			cout << "\t" << aloc[i] << endl;
		
		cout << "z:" << endl;
		for (int i = 0; i < __size - 1; i++)
			cout << "\t" << zloc[i] << endl;
	}

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
				cuda_batch <F, blocks, threads> (
						ins_batched[j],
						outs_batched[j],
						aloc,
						zloc,
						arows,
						zrows,
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

	if (opt) {
		// Deallocate aloc and zloc
		for (int i = 0; i < __size; i++)
			cudaFree(aloc[i]);
		
		for (int i = 0; i < __size - 1; i++)
			cudaFree(zloc[i]);

		delete[] aloc;
		delete[] atloc;
		delete[] zloc;

		delete[] arows;
		delete[] zrows;
	}

	return {t_passed, t_err, t_ktime, t_ftime};
}

}

}

#endif
