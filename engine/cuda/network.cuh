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

// Miscellaneous functions
template <class T>
__host__ __device__
Matrix <T> *NeuralNetwork <T> ::adjusted(T mu)
{
	Matrix <T> *theta = new Matrix <T> [__size - 1];
	for (int i = 0; i < __size - 1; i++)
		theta[i] = __weights[i];

	for (int i = 0; i < __size - 1; i++)
		theta[i] += mu * __momentum[i];

	return theta;
}

template <class T>
__host__ __device__
Matrix <T> *NeuralNetwork <T> ::adjusted(Matrix <T> *weights,
		Matrix <T> *momentum, size_t size, T mu)
{
	Matrix <T> *theta = new Matrix <T> [size - 1];
	for (int i = 0; i < size - 1; i++)
		theta[i] = weights[i];

	for (int i = 0; i < size - 1; i++)
		theta[i] += mu * momentum[i];

	return theta;
}

// Cuda computation functions

// GPU Multithreaded computation
template <class T>
template <size_t blocks, size_t threads>
Vector <T> NeuralNetwork <T> ::cuda_compute(const Vector <T> &in)
{
}

//---------------------------
template <class T>
__host__ __device__
Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
		Vector <T> *a,
		Vector <T> *z) const
{

#ifndef __CUDA_ARCH__

	if (in.size() != __isize)
		throw bad_io_dimensions();

#endif

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < __size - 1) {
		a[i] = tmp.append_above(T (1));

		prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));

		tmp = (*__layers[i + 1].second)(prv);

		Activation <T> *act = __layers[i + 1].second->derivative();

		z[i++] = (*act)(prv);

		delete act;
	}

	a[i] = tmp;
	
	return tmp;
}

template <class T>
__host__ __device__
Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
		Matrix <T> *weights,
		Vector <T> *a,
		Vector <T> *z) const
{

#ifndef __CUDA_ARCH__

	if (in.size() != __isize)
		throw bad_io_dimensions();

#endif

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < __size - 1) {
		a[i] = tmp.append_above(T (1));

		prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

		tmp = (*__layers[i + 1].second)(prv);

		Activation <T> *act = __layers[i + 1].second->derivative();

		z[i++] = (*act)(prv);

		delete act;
	}

	a[i] = tmp;
	
	return tmp;
}

template <class T>
__host__ __device__
Vector <T> NeuralNetwork <T> ::compute_isolated(const Vector <T> &in,
		Matrix <T> *weights,
		Activation <T> **acts,
		Vector <T> *a,
		Vector <T> *z,
		size_t size)
{
	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size - 1) {
		a[i] = tmp.append_above(T (1));

		prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

		Activation <T> *dev_act = copy(acts[i + 1]);

		tmp = (*dev_act)(prv);

		Activation <T> *dev_dact = dev_act->derivative();

		z[i++] = (*dev_dact)(prv);

		delete dev_act;
		delete dev_dact;
	}

	a[i] = tmp;
	
	return tmp;
}

template <class T>
__host__ __device__
Vector <T> NeuralNetwork <T> ::compute_no_cache_isolated(const Vector <T> &in,
		Matrix <T> *weights,
		Activation <T> **acts,
		size_t size)
{
	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size - 1) {
		prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

		tmp = (*acts[i + 1])(prv);
	}
	
	return tmp;
}

// Cuda gradient functions
template <class T>
__host__ __device__
Matrix <T> *NeuralNetwork <T> ::gradient(Matrix <T> *weights,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		bool check)
{

#ifndef __CUDA_ARCH__

	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

#endif

	// Compute the actual value
	Vector <T> actual = compute(in, weights, a, z);
	
	// Get the derivative of the cost
	Optimizer <T> *dopt = opt->derivative();
	
	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [__size - 1];

	Vector <T> delta = (*dopt)(out, actual);
	for (int i = __size - 2; i >= 0; i--) {
		if (i < __size - 2) {
			delta = weights[i + 1].transpose() * delta;
			delta = delta.remove_top();
		}
		
		delta = shur(delta, z[i]);

		Matrix <T> Ji = delta * a[i].transpose();

		J[i] = Ji;
	}

	// Free resources
	delete dopt;

	/* Support gradient checking only for when the function
	 * is called from the host.
	 */

#ifndef __CUDA_ARCH__

	// Skip gradient checking
	if (!check)
		return J;

	// Epsilon value
	T epsilon = 1e-8;

	// Generate individual gradients
	Matrix <T> *qJ = new Matrix <T> [__size - 1];
	for (int i = 0; i < __size - 1; i++)
		qJ[i] = weights[i];
	
	for (int i = 0; i < __size - 1; i++) {
		for (int x = 0; x < weights[i].get_rows(); x++) {
			for (int y = 0; y < weights[i].get_cols(); y++) {
				Matrix <T> *wplus = new Matrix <T> [__size - 1];
				Matrix <T> *wminus = new Matrix <T> [__size - 1];
				
				for (int k = 0; k < __size - 2; k++)
					wplus[k] = wminus[k] = weights[k];

				wplus[i][x][y] += epsilon;
				wminus[i][x][y] -= epsilon;

				Vector <T> jplus = (*opt)(out, compute_no_cache(in, wplus));
				Vector <T> jminus = (*opt)(out, compute_no_cache(in, wminus));

				qJ[i][x][y] = (jplus[0] - jminus[0])/(epsilon + epsilon);

				// Compute the error
				T a = J[i][x][y];
				T b = qJ[i][x][y];

				T d = a - b;

				T e = (d * d) / (a * a + b * b + epsilon);

				// If the error is more than epsilon throw an error
				if (e > epsilon)
					throw bad_gradient();
			}
		}

	}

#endif

	// Return the gradient
	return J;
}

template <class T>
__host__ __device__
Matrix <T> *NeuralNetwork <T> ::gradient_isolated(Matrix <T> *weights,
		Activation <T> **acts,
		size_t size,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt)
{
	// Allocate memory for a and z
	Vector <T> *a = new Vector <T> [size];
	Vector <T> *z = new Vector <T> [size - 1];

	// Compute the actual value
	Vector <T> actual = compute_isolated(
			in,
			weights,
			acts,
			a,
			z,
			size);
	
	// Get the derivative of the cost
	Optimizer <T> *dev_opt = copy(opt);

	Optimizer <T> *dev_dopt = dev_opt->derivative();
	
	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [size - 1];

	Vector <T> delta = (*dev_dopt)(out, actual);
	for (int i = size - 2; i >= 0; i--) {
		if (i < size - 2) {
			delta = weights[i + 1].transpose() * delta;
			delta = delta.remove_top();
		}
		
		delta = shur(delta, z[i]);
		
		Matrix <T> Ji = delta * a[i].transpose();

		J[i] = Ji;
	}

	// Free resources
	delete[] a;
	delete[] z;

	delete dev_opt;
	delete dev_dopt;

	// Return the gradient (skip checking)
	return J;
}

template <class T>
template <class F>
__host__ __device__
void NeuralNetwork <T> ::gradient_and_accumulate_isolated(
		Matrix <T> *weights,
		Activation <T> **acts,
		size_t size,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		F cmp,
		Matrix <T> *grad,
		double &gopt,
		int &gpass)
{
	// Allocate memory for a and z
	Vector <T> *a = new Vector <T> [size];
	Vector <T> *z = new Vector <T> [size - 1];
	
	// Compute the actual value
	Vector <T> actual = compute_isolated(
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

// Cuda training algorithm
template <class T, class F>
__global__ 
void training_kernel(
		Matrix <T> *weights,
		Matrix <T> *momentum,
		size_t *rows,
		size_t *cols,
		Activation <T> **acts,
		Optimizer <T> *opt,
		F cmp,
		size_t net_size,
		T *ins,
		size_t i_size,
		T *outs,
		size_t o_size,
		size_t data_size,
		typename NeuralNetwork <T> ::TrainingStatistics *ts,
		Matrix <T> *Javg,
		Lock lock
	)
{
	// Total number of threads
	int threads = blockDim.x * gridDim.x;
	
	// Thread index
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Thread accumulators
	Matrix <T> *grad = new Matrix <T> [net_size - 1];

	for (int i = 0; i < net_size - 1; i++)
		grad[i] = Matrix <T> (rows[i], cols[i], T(0));

	double gopt = 0.0;
	int gpass = 0;

	// Temporary objects
	Optimizer <T> *dev_opt = copy(opt);

	// Main loop
	for (int i = tid; i < data_size; i += threads) {
		Vector <T> in(i_size, &(ins[i * i_size]));
		Vector <T> out(o_size, &(outs[i * o_size]));
		
		Matrix <T> *adj_weights = NeuralNetwork <T>
			::adjusted(
					weights,
					momentum,
					net_size,
					0.7);
		
		NeuralNetwork <T> ::gradient_and_accumulate_isolated(
					adj_weights,
					acts, 
					net_size,
					in,
					out,
					opt,
					cmp,
					grad,
					gopt,
					gpass);
	}

	delete dev_opt;

	// Combine the gradients into Javg
	lock.lock();

	for (int i = 0; i < net_size - 1; i++)
		Javg[i] += (grad[i] / T(data_size));

	ts->__passed += gpass;
	ts->__cost += gopt;

	lock.unlock();

	delete[] grad;
}

// Training a batch with CUDA
template <class T>
template <class F, size_t blocks, size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::cuda_batch(const DataSet <T> &ins,
		const DataSet <T> &outs,
		T alpha,
		T mu,
		F cmp)
{
	// Timing
	std::chrono::high_resolution_clock::time_point fstart;
	std::chrono::high_resolution_clock::time_point fend;
	
	std::chrono::high_resolution_clock clk;

	// Start timer
	fstart = clk.now();

	// Misc variables
	size_t data_size = ins.size();
	size_t net_size = __size;

	// Flattened IO
	size_t i_size = ins[0].size();
	size_t o_size = outs[0].size();

	size_t i_elems = data_size * i_size;
	size_t o_elems = data_size * o_size;

	T *h_ins = new T[i_elems];
	T *h_outs = new T[o_elems];

	int ki = 0;
	int ko = 0;
	for (int i = 0; i < data_size; i++) {
		for (int j = 0; j < i_size; j++)
			h_ins[ki++] = ins[i][j];
		
		for (int j = 0; j < o_size; j++)
			h_outs[ko++] = outs[i][j];
	}

	// Dimensions
	size_t *h_rows = new size_t[net_size - 1];
	size_t *h_cols = new size_t[net_size - 1];

	for (int i = 0; i < net_size - 1; i++) {
		h_rows[i] = __weights[i].get_rows();
		h_cols[i] = __weights[i].get_cols();
	}

	// Weights
	Matrix <T> *pre_weights = new Matrix <T> [net_size - 1];
	Matrix <T> *pre_momentum = new Matrix <T> [net_size - 1];
	
	for (int i = 0; i < net_size - 1; i++) {
		pre_weights[i].copy_to_device(__weights[i]);
		cudaCheckError(nullptr);
		pre_momentum[i].copy_to_device(__momentum[i]);
		cudaCheckError(nullptr);
	}

	Activation <T> **pre_acts = new Activation <T> *[net_size];

	for (int i = 0; i < net_size; i++) {
		cuda_device_alloc(&pre_acts[i], sizeof(Activation <T>));
		cudaMemcpy(pre_acts[i], __layers[i].second,
				sizeof(Activation <T>),
				cudaMemcpyHostToDevice);
	}

	Matrix <T> *pre_Javg = new Matrix <T> [net_size - 1];

	Matrix <T> tmp;
	for (int i = 0; i < net_size - 1; i++) {
		tmp = __weights[i];
		tmp.set_all(0);

		pre_Javg[i].copy_to_device(tmp);
	}

	// Storage for kernel results
	TrainingStatistics result;

	Matrix <T> *Javg = new Matrix <T> [net_size - 1];

	// Declare all the pointers
	Matrix <T> *dev_weights;
	Matrix <T> *dev_momentum;

	size_t *d_rows;
	size_t *d_cols;

	Activation <T> **dev_acts;
	Optimizer <T> *dev_opt;
	Comparator <T> *dev_cmp;

	T *d_ins;
	T *d_outs;

	TrainingStatistics *dev_result;
	Matrix <T> *dev_Javg;

	// Allocate all the pointers
	cuda_device_alloc(&dev_weights, sizeof(Matrix <T>) * (net_size - 1));
	cuda_device_alloc(&dev_momentum, sizeof(Matrix <T>) * (net_size - 1));
	cuda_device_alloc(&d_rows, sizeof(size_t) * (net_size - 1));
	cuda_device_alloc(&d_cols, sizeof(size_t) * (net_size - 1));
	cuda_device_alloc(&dev_acts, sizeof(Activation <T> *) * net_size);
	cuda_device_alloc(&dev_opt, sizeof(Optimizer <T>));
	cuda_device_alloc(&dev_cmp, sizeof(Comparator <T>));	
	cuda_device_alloc(&d_ins, sizeof(T) * i_elems);
	cuda_device_alloc(&d_outs, sizeof(T) * o_elems);	
	cuda_device_alloc(&dev_result, sizeof(TrainingStatistics));
	cuda_device_alloc(&dev_Javg, sizeof(Matrix <T>) * (net_size - 1));

	// Initialize the data
	cudaMemcpy(dev_weights, pre_weights, sizeof(Matrix <T>) *
			(net_size - 1), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_momentum, pre_momentum, sizeof(Matrix <T>) *
			(net_size - 1), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_rows, h_rows, sizeof(size_t) *
			(net_size - 1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cols, h_cols, sizeof(size_t) *
			(net_size - 1), cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_acts, pre_acts, sizeof(Activation <T> *) *
			net_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_opt, __cost, sizeof(Optimizer <T>),
			cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cmp, &__cmp, sizeof(Comparator <T>),
			cudaMemcpyHostToDevice);

	cudaMemcpy(d_ins, h_ins, sizeof(T) * i_elems,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_outs, h_outs, sizeof(T) * o_elems,
			cudaMemcpyHostToDevice);

	cudaMemcpy(dev_Javg, pre_Javg, sizeof(Vector <T>) *
			(net_size - 1), cudaMemcpyHostToDevice);

	// Run the kernel
	Lock lock;

	cudaEvent_t start;
	cudaEvent_t end;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	training_kernel <<<blocks, threads>>> (
		dev_weights,
		dev_momentum,
		d_rows,
		d_cols,
		dev_acts,
		dev_opt,
		cmp,
		net_size,
		d_ins,
		i_size,
		d_outs,
		o_size,
		data_size,
		dev_result,
		dev_Javg,
		lock
	);
	
	cudaEventRecord(end, 0);

	cudaDeviceSynchronize();
	cudaEventSynchronize(end);

	// Copy items back to host
	cudaMemcpy(&result, dev_result,
			sizeof(TrainingStatistics),
			cudaMemcpyDeviceToHost);
	cudaCheckError(dev_result);

	cudaMemcpy(pre_Javg, dev_Javg, sizeof(Matrix <T>) *
			(net_size - 1), cudaMemcpyDeviceToHost);
	cudaCheckError(dev_Javg);

	float kernel_time;

	cudaEventElapsedTime(&kernel_time, start, end);

	result.__kernel_time = kernel_time;

	// Apply the gradient
	for (int i = 0; i < net_size - 1; i++)
		pre_Javg[i].transfer_from_device(Javg[i]);

	using namespace std;
	cout << "Javg:" << endl;
	for (int i = 0; i < net_size - 1; i++)
		cout << "\t" << Javg[i] << endl;

	// TODO: Uncomment the following line
	// apply_gradient(Javg, alpha, mu);

	// Deallocate memory
	delete[] h_ins;
	delete[] h_outs;
	
	delete[] h_rows;
	delete[] h_cols;

	delete[] pre_weights;
	delete[] pre_momentum;

	delete[] pre_Javg;
	delete[] Javg;

	cudaFree(dev_weights);
	cudaFree(dev_momentum);

	cudaFree(d_rows);
	cudaFree(d_cols);

	cudaFree(dev_acts);
	cudaFree(dev_opt);
	cudaFree(dev_cmp);

	cudaFree(dev_result);
	cudaFree(dev_Javg);

	cudaCheckError(nullptr);

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	for (int i = 0; i < net_size; i++)
		cudaFree(pre_acts[i]);

	delete[] pre_acts;

	// End timer and collect results
	fend = clk.now();
	
	result.__full_time = std::chrono::duration_cast
		<std::chrono::microseconds> (fend - fstart).count();

	// Result the results
	return result;
}

// Epoch training with CUDA
template <class T>
template <class F, size_t blocks, size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::cuda_epochs(const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t iterations,
		size_t batch_size,
		T alpha,
		F cmp,
		bool printing)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

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
		std::cout << std::string(20, '-')
			<< std::endl
			<< "\nEpoch #" << (i + 1)
			<< " (" << lr
			<< ")" << std::endl;
		
		passed = 0;
		err = 0;
		kt = 0;
		ft = 0;

		for (int j = 0; j < ins_batched.size(); j++) {
			TrainingStatistics result =
				cuda_batch <F, blocks, threads> (
						ins_batched[j],
						outs_batched[j],
						lr,
						0.7,
						cmp);

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
		
		std::cout << "\nTotal cost:\t"
			<< err << std::endl
			<< "Kernel time:\t" << kt
			<< " ms" << std::endl
			<< "Full time:\t" << ft/1000
			<< " ms" << std::endl
			<< "Cases passed:\t"
			<< passed
			<< "/" << ins.size() << " ("
			<< 100 * ((double) passed)/ins.size()
			<< "%)" << std::endl;
	}

	return {t_passed, t_err, t_ktime, t_ftime};
}

}

}

#endif
