#ifndef NETWORK_H_
#define NETWORK_H_

// C/C++ headers
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <thread>
#include <vector>

#include <unistd.h>

// JSON library
#include <json/json.hpp>

// Engine headers
#include <core/kernels.hpp>

#include <dataset.hpp>
#include <display.hpp>
#include <layer.hpp>

// #include <optimizer.hpp>

#ifdef ZHP_CUDA

#include <cuda/activation.cuh>
#include <cuda/erf.cuh>
#include <cuda/matrix.cuh>
#include <cuda/vector.cuh>

#else

#include <activation.hpp>
#include <erf.hpp>
#include <matrix.hpp>
#include <vector.hpp>

#endif

// Default path to engine
#ifndef ZHP_ENGINE_PATH

#define ZHP_ENGINE_PATH "engine"

#endif

namespace zhetapi {

namespace ml {

template <class T>
using Comparator = bool (*)(const Vector <T> &, const Vector <T> &);

template <class T>
bool default_comparator(const Vector <T> &a, const Vector <T> &e)
{
	return a == e;
};

// Neural network class
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
	size_t					__size = 0;

	Erf <T> *				__cost = nullptr; // Safe to copy
	Comparator <T>				__cmp = __default_comparator;

	void clear();
public:
	NeuralNetwork();
	NeuralNetwork(const NeuralNetwork &);
	NeuralNetwork(size_t, const std::vector <Layer <T>> &);

	~NeuralNetwork();

	NeuralNetwork &operator=(const NeuralNetwork &);

	/* Saving and loading the network
	void save(const std::string &);
	void load(const std::string &);
	void load_json(const std::string &);

	// Setters
	void set_cost(Erf <T> *);
	void set_comparator(const Comparator <T> &); */

	// Matrix <T> *adjusted(T mu);

	// Computation
	Vector <T> operator()(const Vector <T> &);

	Vector <T> simple_compute(const Vector <T> &);
	
	/*
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
			Erf <T> *,
			bool = false);
	Matrix <T> *gradient(Matrix <T> *,
			const Vector <T> &,
			const Vector <T> &,
			Erf <T> *,
			bool = false);
	Matrix <T> *gradient(Matrix <T> *,
			Vector <T> *,
			Vector <T> *,
			const Vector <T> &,
			const Vector <T> &,
			Erf <T> *,
			bool = false);

	Matrix <T> *simple_gradient(Matrix <T> *,
			const Vector <T> &,
			const Vector <T> &,
			Erf <T> *);
	Matrix <T> *simple_gradient(Matrix <T> *,
			Vector <T> *,
			Vector <T> *,
			const Vector <T> &,
			const Vector <T> &,
			Erf <T> *);

	template <size_t = 1>
	TrainingStatistics validate(
			const DataSet <T> &,
			const DataSet <T> &);

	void train(const Vector <T> &, const Vector <T> &, T);

	template <size_t = 1>
	void simple_train(
			const DataSet <T> &,
			const DataSet <T> &,
			T);

	template <size_t = 1>
	TrainingStatistics train(
			const DataSet <T> &,
			const DataSet <T> &,
			T,
			uint8_t = 0,
			size_t = 0);

	template <size_t = 1>
	TrainingStatistics train_epochs(
			const DataSet <T> &,
			const DataSet <T> &,
			size_t,
			size_t,
			T,
			uint8_t = 0);

	template <size_t = 1>
	TrainingStatistics train_epochs_and_validate(
			const DataSet <T> &,
			const DataSet <T> &,
			const DataSet <T> &,
			const DataSet <T> &,
			size_t,
			size_t,
			T,
			uint8_t = 0);

	void randomize();

	// Printing weights
	void print() const;*/

	static const Comparator <T>		__default_comparator;

	/*
#ifdef ZHP_CUDA

	template <class F>
	TrainingStatistics cuda_batch(
		const DataSet <T> &,
		const DataSet <T> &,
		Activation <T> **,
		Erf <T> *,
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

#endif */

};

// Static variables
template <class T>
const Comparator <T> NeuralNetwork <T> ::__default_comparator
	= default_comparator <T>;

// Constructors and other memory related operations
template <class T>
NeuralNetwork <T> ::NeuralNetwork() {}

template <class T>
NeuralNetwork <T> ::NeuralNetwork(const NeuralNetwork &other) :
		__size(other.__size), __cost(other.__cost),
		__cmp(other.__cmp)
{
	__layers = new Layer <T> [__size];

	for (size_t i = 0; i < __size; i++)
		__layers[i] = other.__layers[i];
}

/*
 * NOTE: The pointers allocated and passed into this function
 * should be left alone. They will be deallocated once the scope
 * of the network object comes to its end. In other words, DO
 * NOT FREE ACTIVATION POINTERS, and instead let the
 * NeuralNetwork class do the work for you.
 */
template <class T>
NeuralNetwork <T> ::NeuralNetwork(size_t isize, const std::vector <Layer <T>> &layers)
		: __size(layers.size())
{
	__layers = new Layer <T> [__size];

	size_t tmp = isize;
	for (int i = 0; i < __size; i++) {
		__layers[i] = layers[i];

		__layers[i].set_fan_in(tmp);

		isize = __layers[i].get_fan_out();
	}
}

template <class T>
NeuralNetwork <T> ::~NeuralNetwork()
{
	clear();
}

template <class T>
NeuralNetwork <T> &NeuralNetwork <T> ::operator=(const NeuralNetwork <T> &other)
{
	if (this != &other) {
		clear();
	}

	return *this;
}

template <class T>
void NeuralNetwork <T> ::clear()
{
	delete[] __layers;
}

// Computation
template <class T>
Vector <T> NeuralNetwork <T> ::operator()(const Vector <T> &in)
{
	return simple_compute(in);
}

template <class T>
Vector <T> NeuralNetwork <T> ::simple_compute(const Vector <T> &in)
{
	Vector <T> tmp = in;

	using namespace std;
	// cout << endl << "in = " << in << endl;
	for (size_t i = 0; i < __size; i++) {
	//	cout << "tmp = " << tmp << endl;
		tmp = __layers[i].forward_propogate(tmp);
	}

	// cout << "tmp = " << tmp << endl;

	return tmp;
}

/*
// Saving and loading
template <class T>
void NeuralNetwork <T> ::save(const std::string &file)
{
	std::ofstream fout(file);

	size_t type_size = sizeof(T);

	fout.write((char *) &type_size, sizeof(size_t));
	fout.write((char *) &__size, sizeof(size_t));
	fout.write((char *) &__isize, sizeof(size_t));
	fout.write((char *) &__osize, sizeof(size_t));

	__layers[0].second->write(fout);
	for (int i = 0; i < __size - 1; i++) {
		size_t r = __weights[i].get_rows();
		size_t c = __weights[i].get_cols();

		fout.write((char *) &r, sizeof(size_t));
		fout.write((char *) &c, sizeof(size_t));

		__weights[i].write(fout);
		__momentum[i].write(fout);
		__layers[i + 1].second->write(fout);
	}
}

template <class T>
void NeuralNetwork <T> ::load(const std::string &file)
{
	std::ifstream fin(file);

	size_t type_size = sizeof(T);

	fin.read((char *) &type_size, sizeof(size_t));
	fin.read((char *) &__size, sizeof(size_t));
	fin.read((char *) &__isize, sizeof(size_t));
	fin.read((char *) &__osize, sizeof(size_t));

	__weights = new Matrix <T> [__size - 1];
	__momentum = new Matrix <T> [__size - 1];
	__layers = new Layer <T> [__size];

	__a = std::vector <Vector <T>> (__size);
	__z = std::vector <Vector <T>> (__size - 1);

	// Read the first activation
	__layers[0].second = Activation <T> ::load(fin);

	// Loop through for the rest
	for (int i = 0; i < __size - 1; i++) {
		size_t r;
		size_t c;

		fin.read((char *) &r, sizeof(size_t));
		fin.read((char *) &c, sizeof(size_t));

		__weights[i] = Matrix <T> (r, c, T(0));
		__momentum[i] = Matrix <T> (r, c, T(0));

		__weights[i].read(fin);
		__momentum[i].read(fin);

		__layers[i + 1].second = Activation <T> ::load(fin);
	}
}

template <class T>
void NeuralNetwork <T> ::load_json(const std::string &file)
{
	std::ifstream fin(file);

	nlohmann::json structure;

	fin >> structure;

	auto layers = structure["Layers"];

	// Allocate size information
	__size = layers.size();
	__isize = layers[0]["Neurons"];
	__osize = layers[__size - 1]["Neurons"];

	// Allocate matrices and activations
	__weights = new Matrix <T> [__size - 1];
	__momentum = new Matrix <T> [__size - 1];
	__layers = new Layer <T> [__size];

	// Allocate caches
	__a = std::vector <Vector <T>> (__size);
	__z = std::vector <Vector <T>> (__size - 1);

	std::vector <size_t> sizes;
	for (size_t i = 0; i < __size; i++) {
		auto layer = layers[i];

		auto activation = layer["Activation"];
		size_t neurons = layer["Neurons"];

		std::vector <T> args;

		for (auto arg : activation["Arguments"])
			args.push_back(arg);

		std::string name = activation["Name"];

		sizes.push_back(layer["Neurons"]);

		__layers[i].second = Activation <T> ::load(name, args);
	}

	for (size_t i = 1; i < __size; i++) {
		__weights[i - 1] = Matrix <T> (sizes[i], sizes[i - 1] + 1, T(0));
		__momentum[i - 1] = Matrix <T> (sizes[i], sizes[i - 1] + 1, T(0));
	}

	__random = default_initializer <T> {};
	__cmp = default_comparator;
}

// Setters
template <class T>
void NeuralNetwork <T> ::set_cost(Erf<T> *opt)
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

	Vector <T> tmp = in;

	size_t i = 0;
	while (i < __size - 1) {
		tmp = __layers[i + 1].second->compute(apt_and_mult(__weights[i], tmp));

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
		Erf <T> *opt,
		bool check)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	// Compute the actual value
	Vector <T> actual = compute(in);

	// Get the derivative of the cost
	Erf <T> *dopt = opt->derivative();

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

	// Return the gradient
	return J;
}

template <class T>
Matrix <T> *NeuralNetwork <T> ::gradient(Matrix <T> *weights,
		const Vector <T> &in,
		const Vector <T> &out,
		Erf <T> *opt,
		bool check)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	// Compute the actual value
	Vector <T> actual = compute(in, weights);

	// Get the derivative of the cost
	Erf <T> *dopt = opt->derivative();

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

	// Return the gradient
	return J;
}

template <class T>
Matrix <T> *NeuralNetwork <T> ::gradient(Matrix <T> *weights,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		const Vector <T> &out,
		Erf <T> *opt,
		bool check)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	// Compute the actual value
	Vector <T> actual = compute(in, weights, a, z);

	// Get the derivative of the cost
	Erf <T> *dopt = opt->derivative();

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

	// Return the gradient
	return J;
}

// No statistical accumulation
template <class T>
Matrix <T> *NeuralNetwork <T> ::simple_gradient(
		Matrix <T> *weights,
		const Vector <T> &in,
		const Vector <T> &out,
		Erf <T> *opt)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	// Compute the actual value
	Vector <T> actual = compute(in, weights);

	// Get the derivative of the cost
	Erf <T> *dopt = opt->derivative();

	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [__size - 1];

	Vector <T> delta = (*dopt)(out, actual);

	for (int i = __size - 2; i >= 0; i--) {
		if (i < __size - 2)
			delta = std::move(rmt_and_mult(weights[i + 1], delta));

		delta.stable_shur(__z[i]);

		J[i] = std::move(vvt_mult(delta, __a[i]));
	}

	// Free resources
	delete dopt;

	// Return the gradient
	return J;
}

template <class T>
Matrix <T> *NeuralNetwork <T> ::simple_gradient(Matrix <T> *weights,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		const Vector <T> &out,
		Erf <T> *opt)
{
	// Check dimensions of input and output
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	// Compute the actual value
	Vector <T> actual = compute(in, weights, a, z);

	// Get the derivative of the cost
	Erf <T> *dopt = opt->derivative();

	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [__size - 1];

	Vector <T> delta = dopt->compute(out, actual);
	for (int i = __size - 2; i >= 0; i--) {
		if (i < __size - 2)
			delta = std::move(rmt_and_mult(weights[i + 1], delta));

		delta.stable_shur(z[i]);

		J[i] = std::move(vvt_mult(delta, a[i]));
	}

	// Free resources
	delete dopt;

	// Return the gradient
	return J;
}

template <class T>
template <size_t threads>
void NeuralNetwork <T> ::simple_train(
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		T alpha)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

	int size = ins.size();

	// Compute modified weights
	Matrix <T> *adj = adjusted(0.7);

	Matrix <T> **grads = new Matrix <T> *[size];
	if (threads == 1) {
		for (int i = 0; i < size; i++)
			grads[i] = simple_gradient(adj, ins[i], outs[i], __cost);
	} else {
		std::vector <std::thread> army;

		auto proc = [&](size_t offset) {
			Vector <T> *aloc = new Vector <T> [__size];
			Vector <T> *zloc = new Vector <T> [__size];

			for (int i = offset; i < size; i += threads)
				grads[i] = simple_gradient(adj, aloc, zloc, ins[i], outs[i], __cost);

			delete[] aloc;
			delete[] zloc;
		};

		for (int i = 0; i < threads; i++)
			army.push_back(std::thread(proc, i));

		for (int i = 0; i < threads; i++)
			army[i].join();
	}

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
	delete[] adj;

	for (int i = 0; i < size; i++)
		delete[] grads[i];

	delete[] grads;
}

template <class T>
void NeuralNetwork <T> ::train(const Vector <T> &in, const Vector <T> &out, T alpha)
{
	Vector <T> actual = compute(in);

	Matrix <T> *adj = adjusted(0.7);
	Matrix <T> *grad = gradient(adj, in, out, __cost);

	delete[] adj;

#ifdef ZHP_GRAD_DEBUG

	using namespace std;
	cout << endl << "Javg:" << endl;
	for (int i = 0; i < __size - 1; i++)
		cout << "\t" << grad[i] << endl;

#else

	apply_gradient(grad, alpha, 0.7);

#endif

	delete[] grad;
}

template <class T>
template <size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::validate(const DataSet <T> &ins,
		const DataSet <T> &outs)
{
	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;

	std::chrono::high_resolution_clock clk;

	size_t passed = 0;
	double opt_error = 0;
	double per_error = 0;

	start = clk.now();

	int size = ins.size();

	if (threads == 1) {
		for (int i = 0; i < size; i++) {
			Vector <T> actual = compute_no_cache(ins[i]);

			if (__cmp(actual, outs[i]))
				passed++;

			opt_error += (*__cost)(outs[i], actual)[0];
			per_error += 100 * (actual - outs[i]).norm()/outs[i].norm();
		}
	} else {
		std::vector <std::thread> army;

		double *optes = new double[threads];
		double *peres = new double[threads];
		int *pass = new int[threads];

		auto proc = [&](size_t offset) {
			for (int i = offset; i < size; i += threads) {
				Vector <T> actual = compute_no_cache(ins[i]);

				if (__cmp(actual, outs[i]))
					pass[offset]++;

				optes[offset] += (*__cost)(outs[i], actual)[0];
				peres[offset] += 100 * (actual - outs[i]).norm()/outs[i].norm();
			}
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

	double tot_time = std::chrono::duration_cast
		<std::chrono::microseconds> (end - start).count();

	return {passed, opt_error, tot_time};
}

template <class T>
template <size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::train(const DataSet <T> &ins,
		const DataSet <T> &outs,
		T alpha,
		uint8_t display,
		size_t id)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

	const int len = 15;
	const int width = 7;

	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;
	std::chrono::high_resolution_clock::time_point total;

	std::chrono::high_resolution_clock clk;

	if (display & Display::batch) {
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
		if (display & Display::batch)
			std::cout << " [";

		for (int i = 0; i < size; i++) {
			Vector <T> actual = compute(ins[i]);

			if (__cmp(actual, outs[i]))
				passed++;

			Matrix <T> *adj = adjusted(0.7);

			grads[i] = gradient(adj, ins[i], outs[i], __cost);

			delete[] adj;

			opt_error += (*__cost)(outs[i], actual)[0];
			per_error += 100 * (actual - outs[i]).norm()/outs[i].norm();

			if (display & Display::batch) {
				int delta = (len * (i + 1))/size;
				for (int i = 0; i < delta - bars; i++) {
					std::cout << "=";
					std::cout.flush();
				}

				bars = delta;
			}
		}

		if (display & Display::batch)
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

	if (display & Display::batch) {
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
	::train_epochs(const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t iterations,
		size_t batch_size,
		T alpha,
		uint8_t display)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

	std::ofstream csv;

	if (display & Display::graph) {
		std::string gpath;

		gpath = "graph.csv";

		csv.open(gpath);

		csv << "epoch,accuracy,loss" << std::endl;

		char pwd_bf[FILENAME_MAX];

		getcwd(pwd_bf, FILENAME_MAX);
		gpath = pwd_bf + ("/" + gpath);

		std::string cmd = "python3 " + (ZHP_ENGINE_PATH + ("/graph/display_graph.py " + gpath)) + " &";
		system(cmd.c_str());
	}

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
	for (size_t i = 0; i < iterations; i++) {
		if (display & Display::epoch) {
			std::cout << std::string(20, '-')
				<< std::endl
				<< "\nEpoch #" << (i + 1)
				<< " (" << lr
				<< ")\n" << std::endl;
		}

		passed = 0;
		err = 0;
		kt = 0;
		for (int j = 0; j < ins_batched.size(); j++) {
			TrainingStatistics result = train <threads> (ins_batched[j],
				outs_batched[j], lr, display, j + 1);

			passed += result.__passed;
			err += result.__cost;
			kt += result.__kernel_time;

			lr = alpha * pow(0.1, (++total)/50000.0);
		}

		t_passed += passed;
		t_err += err;
		t_ktime += kt;

		if (display & Display::epoch) {
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

		if (display & Display::graph) {
			csv << (i + 1) << "," << ((double) passed)/ins.size()
				<< "," << err/ins.size() << std::endl;
		}
	}

	return {t_passed, t_err, t_ktime, t_ftime};
}

template <class T>
template <size_t threads>
typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
	::train_epochs_and_validate(
			const DataSet <T> &ins,
			const DataSet <T> &outs,
			const DataSet <T> &vins,
			const DataSet <T> &vouts,
			size_t iterations,
			size_t batch_size,
			T alpha,
			uint8_t display)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

	std::ofstream csv;

	if (display & Display::graph) {
		std::string gpath;

		gpath = "vgraph.csv";

		csv.open(gpath);

		csv << "epoch,accuracy,loss,vaccuracy,vloss" << std::endl;

		char pwd_bf[FILENAME_MAX];

		getcwd(pwd_bf, FILENAME_MAX);
		gpath = pwd_bf + ("/" + gpath);

		std::string cmd = "python3 " + (ZHP_ENGINE_PATH + ("/graph/display_graph_w_valid.py " + gpath)) + " &";
		system(cmd.c_str());
	}

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
	for (size_t i = 0; i < iterations; i++) {
		if (display & Display::epoch) {
			std::cout << std::string(20, '-')
				<< std::endl
				<< "\nEpoch #" << (i + 1)
				<< " (" << lr
				<< ")\n" << std::endl;
		}

		passed = 0;
		err = 0;
		kt = 0;
		for (int j = 0; j < ins_batched.size(); j++) {
			TrainingStatistics result = train <threads> (ins_batched[j],
				outs_batched[j], lr, display, j + 1);

			passed += result.__passed;
			err += result.__cost;
			kt += result.__kernel_time;

			lr = alpha * pow(0.1, (++total)/50000.0);
		}

		t_passed += passed;
		t_err += err;
		t_ktime += kt;

		if (display & Display::epoch) {
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

		if (display & Display::graph) {
			TrainingStatistics ts = validate <threads> (vins, vouts);

			csv << (i + 1) << "," << ((double) passed)/ins.size()
				<< "," << err/ins.size() << ","
				<< ((double) ts.__passed)/vins.size() << ","
				<< ts.__cost/vins.size() << std::endl;
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
} */

}

}

#endif
