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

template <class T>
__host__ __device__
Matrix <T> *adjusted(
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
Vector <T> compute_isolated_parallelized(
		const Vector <T> &in,
		Matrix <T> *weights,
		ml::Activation <T> **acts,
		Vector <T> *a,
		Vector <T> *z,
		size_t size)
{
	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size - 1) {
		a[i] = tmp.append_above(T (1));

		prv = weights[i] * a[i];

		ml::Activation <T> *dev_act = copy(acts[i + 1]);

		tmp = (*dev_act)(prv);

		ml::Activation <T> *dev_dact = dev_act->derivative();

		z[i++] = (*dev_dact)(prv);

		delete dev_act;
		delete dev_dact;
	}

	a[i] = tmp;
	
	return tmp;
}

template <class T, class F>
void gradient_and_accumulate_isolated_parallelized(
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
	Vector <T> actual = compute_isolated_parallelized(
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

// Training a batch with CUDA
template <class T>
template <class F, size_t blocks, size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::cuda_batch(const DataSet <T> &ins,
		const DataSet <T> &outs,
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
	
	size_t data_size = ins.size();
	size_t net_size = __size;

	Matrix <T> *J = new Matrix <T> [net_size - 1];
	for (int i = 0; i < net_size - 1; i++) {
		J[i] = Matrix <T> (__weights[i].get_rows(),
				__weights[i].get_cols(), T(0));
	}
	
	size_t elems = ins.size();
	size_t i_size = ins[0].size();
	size_t o_size = outs[0].size();

	Activation <T> **acts = new Activation <T> *[__size];
	for (int i = 0; i < __size; i++)
		acts[i] = __layers[i].second;

	double gopt = 0;
	size_t gpass = 0;

	// Start kernel timer
	lstart = clk.now();
	
	Matrix <T> *adj_weights = zhetapi::ml::adjusted(
				__weights,
				__momentum,
				net_size,
				0.7);

	for (int i = 0; i < data_size; i++) {
		Vector <T> in = ins[i];
		Vector <T> out = outs[i];

		gradient_and_accumulate_isolated_parallelized(
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

	return {t_passed, t_err, t_ktime, t_ftime};
}

}

}

#endif
