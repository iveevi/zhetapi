#ifndef NETWORK_H_
#define NETWORK_H_

// C/C++ headers
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <thread>
#include <vector>

// JSON library
#include <json.hpp>

// Engine headers
#include <dataset.hpp>

#ifndef ZHP_CUDA

#include <activation.hpp>
#include <matrix.hpp>
#include <optimizer.hpp>
#include <vector.hpp>

#else

#include <cuda/activation.cuh>
#include <cuda/matrix.cuh>
#include <cuda/optimizer.cuh>
#include <cuda/vector.cuh>

#endif

namespace zhetapi {
		
namespace ml {

template <class T>
using Layer = std::pair <size_t, Activation <T> *>;

template <class T>
using Comparator = bool (*)(const Vector <T> &, const Vector <T> &);

template <class T>
bool default_comparator(const Vector <T> &a, const Vector <T> &e)
{
	return a == e;
};

#ifdef ZHP_CUDA

template <class T>
__device__
bool cuda_default_comparator(const Vector <T> &a, const Vector <T> &e)
{
	return a == e;
};

#endif

/*
* Nerual Network
*
* @tparam T is the type with which calculations are performed
* @tparam U is the type of activation parameter scheme, ie. unary or
* binary
*/
template <class T>
class NeuralNetwork {
public:
	// Training statistics
	struct TrainingStatistics {
		size_t	__passed = 0;
		T	__cost = T(0);
		double	__kernel_time = 0;
		double	__full_time = 0;
	};

	// Exceptions
	class bad_gradient {};
	class bad_io_dimensions {};
private:
	Layer <T> *				__layers = nullptr;
	Matrix <T> *				__weights = nullptr;
	Matrix <T> *				__momentum = nullptr;
	std::function <T ()>			__random;
	size_t					__isize = 0;
	size_t					__osize = 0;
	size_t					__size = 0;

	std::vector <Vector <T>>		__a = {};
	std::vector <Vector <T>>		__z = {};

	Optimizer <T> *				__cost = nullptr;
	
	Comparator <T>				__cmp = __default_comparator;
public:
	NeuralNetwork();
	NeuralNetwork(const std::vector <Layer <T>> &, const std::function <T ()> &);

	~NeuralNetwork();

	// Saving and loading the network
	void save(const std::string &);
	void load_json(const std::string &);

	// Setters
	void set_cost(Optimizer <T> *);
	void set_comparator(const Comparator <T> &);

	Matrix <T> *adjusted(T mu);
	
	Vector <T> compute(const Vector <T> &);
	Vector <T> compute(const Vector <T> &,
			Matrix <T> *);
	Vector <T> compute(const Vector <T> &,
			Vector <T> *,
			Vector <T> *) const;
	Vector <T> compute(const Vector <T> &,
			Matrix <T> *,
			Vector <T> *,
			Vector <T> *) const;
	
	Vector <T> compute_no_cache(const Vector <T> &) const;
	Vector <T> compute_no_cache(const Vector <T> &,
			Matrix <T> *) const;

	Vector <T> operator()(const Vector <T> &);

	void apply_gradient(Matrix <T> *, T, T);

	Matrix <T> *gradient(const Vector <T> &,
			const Vector <T> &,
			Optimizer <T> *,
			bool = false);
	Matrix <T> *gradient(Matrix <T> *,
			const Vector <T> &,
			const Vector <T> &,
			Optimizer <T> *,
			bool = false);
	Matrix <T> *gradient(Matrix <T> *,
			Vector <T> *,
			Vector <T> *,
			const Vector <T> &,
			const Vector <T> &,
			Optimizer <T> *,
			bool = false);

	template <size_t = 1>
	TrainingStatistics train(const DataSet <T> &,
		const DataSet <T> &,
		T, size_t = 0,
		bool = false);

	template <size_t = 1>
	TrainingStatistics epochs(const DataSet <T> &,
		const DataSet <T> &,
		size_t, size_t, T,
		bool = false);

	void randomize();

	// Printing weights
	void print() const;

	static const Comparator <T>		__default_comparator;

#ifdef ZHP_CUDA

	template <class F>
	TrainingStatistics cuda_batch(
		const DataSet <T> &,
		const DataSet <T> &,
		Activation <T> **,
		Optimizer <T> *,
		T *,
		T *,
		T *,
		T *,
		size_t *,
		size_t *,
		size_t *,
		size_t *,
		size_t,
		T **,
		T **,
		T **,
		T **,
		size_t *,
		size_t *,
		size_t *,
		T,
		T,
		F,
		bool = false);

	template <class F>
	TrainingStatistics cuda_epochs(
		const DataSet <T> &,
		const DataSet <T> &,
		size_t,
		size_t,
		T,
		F,
		bool = false);

#endif

};

// Static variables
template <class T>
const Comparator <T> NeuralNetwork <T> ::__default_comparator
	= default_comparator <T>;

// Constructors
template <class T>
NeuralNetwork <T> ::NeuralNetwork() {}

/*
 * NOTE: The pointers allocated and passed into this function
 * should be left alone. They will be deallocated once the scope
 * of the network object comes to its end. In other words, DO
 * NOT FREE ACTIVATION POINTERS, and instead let the
 * NeuralNetwork class do the work for you.
 */
template <class T>
NeuralNetwork <T> ::NeuralNetwork(const std::vector <Layer <T>> &layers,
		const std::function <T ()> &random)
		: __random(random),
		__size(layers.size()),
		__isize(layers[0].first),
		__osize(layers[layers.size() - 1].first),
		__layers(nullptr),
		__cost(nullptr),
		__cmp(__default_comparator)
{
	__layers = new Layer <T> [__size];
	for (int i = 0; i < __size; i++)
		__layers[i] = layers[i];
	
	__weights = new Matrix <T> [__size - 1];
	__momentum = new Matrix <T> [__size - 1];

	__a = std::vector <Vector <T>> (__size);
	__z = std::vector <Vector <T>> (__size - 1);
	for (size_t i = 0; i < __size - 1; i++) {
		// Add extra column for constants (biases)
		Matrix <T> mat(__layers[i + 1].first, __layers[i].first + 1);

		__weights[i] = mat;
		__momentum[i] = mat;
	}
}

template <class T>
NeuralNetwork <T> ::~NeuralNetwork()
{
	for (int i = 0; i < __size; i++)
		delete __layers[i].second;

	delete[] __layers;
	delete[] __weights;
	delete[] __momentum;
}

// Saving and loading
template <class T>
void NeuralNetwork <T> ::save(const std::string &file)
{
	std::ofstream fout(file);

	size_t type_size = sizeof(T);

	fout.write((char *) &type_size, sizeof(size_t));
	fout.write((char *) &__size, sizeof(size_t));

	for (int i = 0; i < __size - 1; i++) {
		size_t r = __weights[i].get_rows();
		size_t c = __weights[i].get_cols();

		fout.write((char *) &r, sizeof(size_t));
		fout.write((char *) &c, sizeof(size_t));

		for (int k = 0; k < r * c; k++)
			fout.write((char *) __weights[i][k], sizeof(T));
	}
}

template <class T>
void NeuralNetwork <T> ::load_json(const std::string &file)
{
	std::ifstream fin(file);

	nlohmann::json structure;

	fin >> structure;

	std::cout << "Parsed structure: " << std::endl << structure << std::endl;
	std::cout << "elem0: " << structure["Layers"] << std::endl;

	auto layers = structure["Layers"];
	std::cout << "type: " << layers.type_name() << std::endl;
	std::cout << "size: " << layers.size() << std::endl;

	size_t n = layers.size();

	for (int i = 0; i < n; i++) {
		auto layer = layers[i];

		std::string activation = layer["Activation"];
		size_t neurons = layer["Neurons"];

		std::cout << "layers[" << i << "]:" << std::endl;
		std::cout << "\t" << activation << std::endl;
		std::cout << "\t" << neurons << std::endl;
	}
}

// Setters
template <class T>
void NeuralNetwork <T> ::set_cost(Optimizer<T> *opt)
{
	__cost = opt;
}

template <class T>
void NeuralNetwork <T> ::set_comparator(const Comparator <T> &cmp)
{
	__cmp = cmp;
}

template <class T>
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
Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in)
{
	if (in.size() != __isize)
		throw bad_io_dimensions();

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < __size - 1) {
		__a[i] = tmp.append_above(T (1));

		prv = __weights[i] * tmp.append_above(T (1));

		tmp = (*__layers[i + 1].second)(prv);

		Activation <T> *act = __layers[i + 1].second->derivative();

		__z[i++] = (*act)(prv);

		delete act;
	}

	__a[i] = tmp;
	
	return tmp;
}

template <class T>
Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
		Matrix <T> *weights)
{
	if (in.size() != __isize)
		throw bad_io_dimensions();

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < __size - 1) {
		__a[i] = tmp.append_above(T (1));

		prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

		tmp = (*__layers[i + 1].second)(prv);

		Activation <T> *act = __layers[i + 1].second->derivative();

		__z[i++] = (*act)(prv);

		delete act;
	}

	__a[i] = tmp;
	
	return tmp;
}

template <class T>
Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
		Vector <T> *a,
		Vector <T> *z) const
{
	if (in.size() != __isize)
		throw bad_io_dimensions();

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
Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
		Matrix <T> *weights,
		Vector <T> *a,
		Vector <T> *z) const
{
	if (in.size() != __isize)
		throw bad_io_dimensions();

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
Vector <T> NeuralNetwork <T> ::compute_no_cache(const Vector <T> &in) const
{
	if (in.size() != __isize)
		throw bad_io_dimensions();

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < __size - 1) {
		prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));

		tmp = (*__layers[i + 1].second)(prv);

		Activation <T> *act = __layers[i + 1].second->derivative();

		delete act;

		i++;
	}
	
	return tmp;
}

template <class T>
Vector <T> NeuralNetwork <T> ::compute_no_cache(const Vector <T> &in,
		Matrix <T> *weights) const
{
	if (in.size() != __isize)
		throw bad_io_dimensions();

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < __size - 1) {
		prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

		tmp = (*__layers[i + 1].second)(prv);

		Activation <T> *act = __layers[i + 1].second->derivative();

		delete act;

		i++;
	}
	
	return tmp;
}

template <class T>
Vector <T> NeuralNetwork <T> ::operator()(const Vector <T> &in)
{
	return compute(in);
}

template <class T>
void NeuralNetwork <T> ::apply_gradient(Matrix <T> *grad,
		T alpha,
		T mu)
{
	for (int i = 0; i < __size - 1; i++) {
		__momentum[i] = mu * __momentum[i] - alpha * grad[i];
		__weights[i] += __momentum[i];
	}
}

template <class T>
Matrix <T> *NeuralNetwork <T> ::gradient(const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		bool check)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	// Compute the actual value
	Vector <T> actual = compute(in);
	
	// Get the derivative of the cost
	Optimizer <T> *dopt = opt->derivative();
	
	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [__size - 1];

	Vector <T> delta = (*dopt)(out, actual);
	for (int i = __size - 2; i >= 0; i--) {
		if (i < __size - 2) {
			delta = __weights[i + 1].transpose() * delta;
			delta = delta.remove_top();
		}
		
		delta = shur(delta, __z[i]);

		Matrix <T> Ji = delta * __a[i].transpose();

		J[i] = Ji;
	}

	// Free resources
	delete dopt;

	// Skip gradient checking
	if (!check)
		return J;

	// Epsilon value
	T epsilon = 1e-8;

	// Generate individual gradients
	Matrix <T> *qJ = new Matrix <T> [__size - 1];
	for (int i = 0; i < __size - 1; i++)
		qJ[i] = __weights[i];
	
	for (int i = 0; i < __size - 1; i++) {
		for (int x = 0; x < __weights[i].get_rows(); x++) {
			for (int y = 0; y < __weights[i].get_cols(); y++) {
				Matrix <T> *wplus = new Matrix <T> [__size - 1];
				Matrix <T> *wminus = new Matrix <T> [__size - 1];
				
				for (int k = 0; k < __size - 1; k++)
					wplus[k] = wminus[k] = __weights[k];

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

	// Return the gradient
	return J;
}

template <class T>
Matrix <T> *NeuralNetwork <T> ::gradient(Matrix <T> *weights,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		bool check)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	// Compute the actual value
	Vector <T> actual = compute(in, weights);
	
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
		
		delta = shur(delta, __z[i]);

		Matrix <T> Ji = delta * __a[i].transpose();

		J[i] = Ji;
	}

	// Free resources
	delete dopt;

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
				
				for (int k = 0; k < __size - 1; k++)
					wplus[k] = wminus[k] = __weights[k];

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

	// Return the gradient
	return J;
}

template <class T>
Matrix <T> *NeuralNetwork <T> ::gradient(Matrix <T> *weights,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		const Vector <T> &out,
		Optimizer <T> *opt,
		bool check)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

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

	// Return the gradient
	return J;
}

template <class T>
template <size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::train(const DataSet <T> &ins,
		const DataSet <T> &outs,
		T alpha,
		size_t id,
		bool printing)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();
	
	const int len = 15;
	const int width = 7;

	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;
	std::chrono::high_resolution_clock::time_point total;
	
	std::chrono::high_resolution_clock clk;

	if (printing) {
		std::string str = "#" + std::to_string(id);

		std::cout << "Batch " << std::setw(6)
			<< str << " (" << ins.size() << ")";
	}

	size_t passed = 0;

	int bars = 0;

	double opt_error = 0;
	double per_error = 0;

	start = clk.now();

	int size = ins.size();

	using namespace std;

	Matrix <T> **grads = new Matrix <T> *[size];
	if (threads == 1) {
		if (printing)
			std::cout << " [";
		
		for (int i = 0; i < size; i++) {
			Vector <T> actual = compute(ins[i]);

			if (__cmp(actual, outs[i]))
				passed++;

			/* The gradient function allocates the
			 * memory anyways, no need to allocate here.
			  */
			Matrix <T> *adj = adjusted(0.7);

			grads[i] = gradient(adj, ins[i], outs[i], __cost);
			
			/* using namespace std;
			cout << endl << "grad[i] = " << endl;
			for (int j = 0; j < __size - 1; j++)
				cout << "\t" << grads[i][j] << endl; */

			delete[] adj;

			opt_error += (*__cost)(outs[i], actual)[0];
			per_error += 100 * (actual - outs[i]).norm()/outs[i].norm();

			if (printing) {
				int delta = (len * (i + 1))/size;
				for (int i = 0; i < delta - bars; i++) {
					std::cout << "=";
					std::cout.flush();
				}

				bars = delta;
			}
		}

		if (printing)
			std::cout << "]";
	} else {
		std::vector <std::thread> army;
		
		double *optes = new double[threads];
		double *peres = new double[threads];
		int *pass = new int[threads];

		auto proc = [&](size_t offset) {
			Vector <T> *aloc = new Vector <T> [__size];
			Vector <T> *zloc = new Vector <T> [__size];

			for (int i = offset; i < size; i += threads) {
				Vector <T> actual = compute(ins[i], aloc, zloc);

				// TODO: Determine whether the the
				// following if statement is
				// hazardous.
				if (__cmp(actual, outs[i]))
					pass[offset]++;

				Matrix <T> *adj = adjusted(0.7);

				grads[i] = gradient(adj, aloc, zloc, ins[i], outs[i], __cost);

				delete[] adj;
			
				optes[offset] += (*__cost)(outs[i], actual)[0];
				peres[offset] += 100 * (actual - outs[i]).norm()/outs[i].norm();
			}

			delete[] aloc;
			delete[] zloc;
		};

		for (int i = 0; i < threads; i++) {
			optes[i] = peres[i] = pass[i] = 0;

			army.push_back(std::thread(proc, i));
		}

		for (int i = 0; i < threads; i++) {
			army[i].join();

			opt_error += optes[i];
			per_error += peres[i];
			passed += pass[i];
		}

		// Free resources
		delete[] optes;
		delete[] peres;
		delete[] pass;
	}

	end = clk.now();

	Matrix <T> *grad = new Matrix <T> [__size - 1];
	for (int i = 0; i < __size - 1; i++)
		grad[i] = grads[0][i];
	
	for (size_t i = 1; i < size; i++) {
		for (size_t j = 0; j < __size - 1; j++)
			grad[j] += grads[i][j];
	}

	for (size_t j = 0; j < __size - 1; j++)
		grad[j] /= (double) size;

#ifdef ZHP_GRAD_DEBUG

	using namespace std;
	cout << endl << "Javg:" << endl;
	for (int i = 0; i < __size - 1; i++)
		cout << "\t" << grad[i] << endl;

#else

	apply_gradient(grad, alpha, 0.7);

#endif
	
	// Release memory
	delete[] grad;

	for (int i = 0; i < size; i++)
		delete[] grads[i];

	delete[] grads;
	
	// Stop timer
	total = clk.now();

	double avg_time = std::chrono::duration_cast
		<std::chrono::microseconds> (end - start).count();
	double tot_time = std::chrono::duration_cast
		<std::chrono::microseconds> (total - start).count();
	avg_time /= size;

	if (printing) {
		std::cout << " passed: " << passed << "/" << size << " = "
			<< std::fixed << std::showpoint << std::setprecision(2)
			<< 100 * ((double) passed)/size << "%, "
			<< "µ-err: "
			<< std::setw(width) << std::fixed
			<< std::showpoint << std::setprecision(2)
			<< per_error/size << "%, "
			<< "µ-time: " << avg_time << " µs"
			<< std::endl;
	}
	
	return {passed, opt_error, tot_time};
}

template <class T>
template <size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::epochs(const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t iterations,
		size_t batch_size,
		T alpha,
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
	T err;
	for (int i = 0; i < iterations; i++) {
		if (printing) {
			std::cout << std::string(20, '-')
				<< std::endl
				<< "\nEpoch #" << (i + 1)
				<< " (" << lr
				<< ")" << std::endl;
		}

		if (printing)
			std::cout << std::endl;
		
		passed = 0;
		err = 0;
		kt = 0;
		for (int j = 0; j < ins_batched.size(); j++) {
			TrainingStatistics result = train <threads> (ins_batched[j],
				outs_batched[j], lr, j + 1, printing);

			passed += result.__passed;
			err += result.__cost;
			kt += result.__kernel_time;

			lr = alpha * pow(0.1, (++total)/50000.0);
		}

		t_passed += passed;
		t_err += err;
		t_ktime += kt;
		
		if (printing) {
			std::cout << "\nTotal cost:\t"
				<< err << std::endl
				<< "Total time:\t" << kt/1000
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

template <class T>
void NeuralNetwork <T> ::randomize()
{
	for (int i = 0; i < __size - 1; i++)			
		__weights[i].randomize(__random);
}

template <class T>
void NeuralNetwork <T> ::print() const
{
	std::cout << "================================" << std::endl;
	
	std::cout << "Weights:" << std::endl;

	size_t n = 0;
	for (int i = 0; i < __size - 1; i++)
		std::cout << "[" << ++n << "]\t" << __weights[i] << std::endl;
	
	std::cout << "Momentum:" << std::endl;

	n = 0;
	for (int i = 0; i < __size - 1; i++)
		std::cout << "[" << ++n << "]\t" << __momentum[i] << std::endl;
	
	std::cout << "================================" << std::endl;
}

}

}

#endif
