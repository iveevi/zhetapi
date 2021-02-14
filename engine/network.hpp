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
#include <gradient.hpp>
#include <optimizer.hpp>

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

// Neural network class
template <class T = double>
class NeuralNetwork {
public:
	// Exceptions
	class bad_io_dimensions {};
	class null_optimizer {};
private:
	Layer <T> *		__layers	= nullptr;

	size_t			__size		= 0;
	size_t			__isize		= 0;
	size_t			__osize		= 0;

	// Remove erf and optimizer later
	Erf <T> *		__cost		= nullptr; // Safe to copy
	Optimizer <T> *         __opt		= nullptr;

	void clear();
public:
	NeuralNetwork();
	NeuralNetwork(const NeuralNetwork &);
	NeuralNetwork(size_t, const std::vector <Layer <T>> &);

	~NeuralNetwork();

	NeuralNetwork &operator=(const NeuralNetwork &);

	// Saving and loading the network
	void save(const std::string &);
	void load(const std::string &);
	// void load_json(const std::string &);

	// Setters
	void set_cost(Erf <T> *);
	void set_optimizer(Optimizer <T> *);

	void diagnose() const;
	void print() const;

	// Computation
	Vector <T> operator()(const Vector <T> &);

	Vector <T> compute(const Vector <T> &);
	
	void fit(const Vector <T> &, const Vector <T> &);
	void fit(const DataSet <T> &, const DataSet <T> &);	
	
	void multithreaded_fit(const DataSet <T> &, const DataSet <T> &, size_t = 1);	
};

// Constructors and other memory related operations
template <class T>
NeuralNetwork <T> ::NeuralNetwork() {}

template <class T>
NeuralNetwork <T> ::NeuralNetwork(const NeuralNetwork &other) :
		__size(other.__size), __isize(other.__isize),
		__osize(other.__osize), __cost(other.__cost),
		__opt(other.__opt)
{
	__layers = new Layer <T> [__size];

	for (size_t i = 0; i < __size; i++)
		__layers[i] = other.__layers[i];
}

/*
 * TODO: Add a input class from specifying input format.
 * NOTE: The pointers allocated and passed into this function
 * should be left alone. They will be deallocated once the scope
 * of the network object comes to its end. In other words, DO
 * NOT FREE ACTIVATION POINTERS, and instead let the
 * NeuralNetwork class do the work for you.
 */
template <class T>
NeuralNetwork <T> ::NeuralNetwork(size_t isize, const std::vector <Layer <T>> &layers)
		: __size(layers.size()), __isize(isize)
{
	__layers = new Layer <T> [__size];

	size_t tmp = isize;
	for (int i = 0; i < __size; i++) {
		__layers[i] = layers[i];

		__layers[i].set_fan_in(tmp);

		__layers[i].initialize();

		tmp = __layers[i].get_fan_out();
	}

	__osize = tmp;
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

		__size = other.__size;
		__isize = other.__isize;
		__osize = other.__osize;

		__cost = other.__cost;
		__opt = other.__opt;

		__layers = new Layer <T> [__size];
		for (size_t i = 0; i < __size; i++)
			__layers[i] = other.__layers[i];
	}

	return *this;
}

template <class T>
void NeuralNetwork <T> ::clear()
{
	delete[] __layers;
}

// Saving and loading
template <class T>
void NeuralNetwork <T> ::save(const std::string &file)
{
	std::ofstream fout(file);

	fout.write((char *) &__size, sizeof(size_t));
	fout.write((char *) &__isize, sizeof(size_t));
	fout.write((char *) &__osize, sizeof(size_t));

	for (int i = 0; i < __size; i++)
		__layers[i].write(fout);
}

template <class T>
void NeuralNetwork <T> ::load(const std::string &file)
{
	// Clear the current members
	clear();

	std::ifstream fin(file);

	fin.read((char *) &__size, sizeof(size_t));
	fin.read((char *) &__isize, sizeof(size_t));
	fin.read((char *) &__osize, sizeof(size_t));

	__layers = new Layer <T> [__size];
	for (size_t i = 0; i < __size; i++)
		__layers[i].read(fin);
}

/*
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
} */

// Property managers and viewers
template <class T>
void NeuralNetwork <T> ::set_cost(Erf <T> *cost)
{
	__cost = cost;
}

template <class T>
void NeuralNetwork <T> ::set_optimizer(Optimizer <T> *opt)
{
	__opt = opt;
}

// Computation
template <class T>
Vector <T> NeuralNetwork <T> ::operator()(const Vector <T> &in)
{
	return compute(in);
}

template <class T>
Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in)
{
	Vector <T> tmp = in;

	for (size_t i = 0; i < __size; i++)
		tmp = __layers[i].forward_propogate(tmp);

	return tmp;
}

template <class T>
void NeuralNetwork <T> ::fit(const Vector <T> &in, const Vector <T> &out)
{
	if ((in.size() != __isize) || (out.size() != __osize))
		throw bad_io_dimensions();

	if (!__opt)
		throw null_optimizer();

	Matrix <T> *J = __opt->gradient(__layers, __size, in, out, __cost);

	for (size_t i = 0; i < __size; i++)
		__layers[i].apply_gradient(J[i]);

	delete[] J;
}

template <class T>
void NeuralNetwork <T> ::fit(const DataSet <T> &ins, const DataSet <T> &outs)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

	if ((ins[0].size() != __isize) || (outs[0].size() != __osize))
		throw bad_io_dimensions();

	if (!__opt)
		throw null_optimizer();

	Matrix <T> *J = __opt->batch_gradient(__layers, __size, ins, outs, __cost);

	for (size_t i = 0; i < __size; i++)
		__layers[i].apply_gradient(J[i]);

	delete[] J;
}

template <class T>
void NeuralNetwork <T> ::multithreaded_fit(
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t threads)
{
	if (ins.size() != outs.size())
		throw bad_io_dimensions();

	if ((ins[0].size() != __isize) || (outs[0].size() != __osize))
		throw bad_io_dimensions();

	if (!__opt)
		throw null_optimizer();

	Matrix <T> *J = __opt->multithreaded_batch_gradient(
			__layers,
			__size,
			ins,
			outs,
			__cost,
			threads);

	for (size_t i = 0; i < __size; i++)
		__layers[i].apply_gradient(J[i]);

	delete[] J;
}

template <class T>
void NeuralNetwork <T> ::diagnose() const
{
	for (size_t i = 0; i < __size; i++)
		__layers[i].diagnose();
}

template <class T>
void NeuralNetwork <T> ::print() const
{
	for (size_t i = 0; i < __size; i++)
		__layers[i].print();
}

}

}

#endif
