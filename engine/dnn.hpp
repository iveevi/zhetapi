#ifndef DNN_H_
#define DNN_H_

#ifndef __AVR	// Does not support AVR

// C/C++ headers
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
// #include <json/json.hpp>

// Engine headers
#include "dataset.hpp"

#endif		// Does not support AVR

// Engine headers
#include "core/kernels.hpp"

#include "display.hpp"
#include "layer.hpp"
#include "gradient.hpp"
#include "optimizer.hpp"

#ifdef ZHP_CUDA

#include <cuda/activation.cuh>
#include <cuda/erf.cuh>
#include <cuda/matrix.cuh>
#include <cuda/vector.cuh>

#else

#include "activation.hpp"
#include "erf.hpp"
#include "matrix.hpp"
#include "vector.hpp"

#endif

// Default path to engine
#ifndef ZHP_ENGINE_PATH

#define ZHP_ENGINE_PATH "engine"

#endif

namespace zhetapi {

namespace ml {

// Type aliases
template <class T>
using DNNGrad = Vector <Matrix <T>>;

/**
 * @brief Neural network class
 */
template <class T = double>
class DNN {
public:
	// Exceptions
	class bad_io_dimensions {};
	class null_optimizer {};
	class null_loss_function {};
private:
	Layer <T> *		_layers	= nullptr;

	size_t			_size		= 0;
	size_t			_isize		= 0;
	size_t			_osize		= 0;

	Vector <T> *		_acache	= nullptr;
	Vector <T> *		_zcache	= nullptr;

	Matrix <T> *		_Acache	= nullptr;
	Matrix <T> *		_Zcache	= nullptr;

	// Variadic constructor helper
	size_t fill(size_t, const Layer <T> &);

	template <class ... U>
	size_t fill(size_t, const Layer <T> &, U ...);

	// Other private helpers
	void clear();
	void clear_cache();
	void init_cache();
public:
	DNN();
	DNN(const DNN &);
	
	AVR_IGNORE(DNN(size_t, const std::vector <Layer <T>> &));

	// Variadic layer constructor
	template <class ... U>
	DNN(size_t, size_t, U ...);

	~DNN();

	DNN &operator=(const DNN &);

	// Saving and loading the network
	AVR_IGNORE(void save(const std::string &));
	AVR_IGNORE(void load(const std::string &));
	
	// void load_json(const std::string &);
	
	// Properties
	size_t size() const;
	size_t input_size() const;
	size_t output_size() const;
	
	Vector <T> *acache() const;
	Vector <T> *zcache() const;

	Layer <T> *layers();

	// Miscellaneous
	void print() const;

	// Computation
	Vector <T> operator()(const Vector <T> &);

	Vector <T> compute(const Vector <T> &);
	
	void apply_gradient(Matrix <T> *);

	Matrix <T> *jacobian(const Vector <T> &);
	Matrix <T> *jacobian_delta(const Vector <T> &, Vector <T> &);
};

// Constructors and other memory related operations
template <class T>
DNN <T> ::DNN() {}

template <class T>
DNN <T> ::DNN(const DNN &other) :
		_size(other._size), _isize(other._isize),
		_osize(other._osize)
{
	_layers = new Layer <T> [_size];

	for (size_t i = 0; i < _size; i++)
		_layers[i] = other._layers[i];
	
	init_cache();
}

// TODO: new file
#ifndef __AVR	// Does not support AVR

/*
 * TODO: Add a input class from specifying input format.
 * NOTE: The pointers allocated and passed into this function
 * should be left alone. They will be deallocated once the scope
 * of the network object comes to its end. In other words, DO
 * NOT FREE ACTIVATION POINTERS, and instead let the
 * DNN class do the work for you.
 */
template <class T>
DNN <T> ::DNN(size_t isize, const std::vector <Layer <T>> &layers)
		: _size(layers.size()), _isize(isize)
{
	_layers = new Layer <T> [_size];

	size_t tmp = isize;
	for (int i = 0; i < _size; i++) {
		_layers[i] = layers[i];

		_layers[i].set_fan_in(tmp);

		_layers[i].initialize();

		tmp = _layers[i].get_fan_out();
	}

	_osize = tmp;

	init_cache();
}

#endif		// Does not support AVR

template <class T>
template <class ... U>
DNN <T> ::DNN(size_t layers, size_t isize, U ... args)
		: _size(layers), _isize(isize)
{
	_layers = new Layer <T> [_size];

	_osize = fill(0, args...);

	init_cache();
}

template <class T>
DNN <T> ::~DNN()
{
	clear();
}

template <class T>
DNN <T> &DNN <T> ::operator=(const DNN <T> &other)
{
	if (this != &other) {
		clear();

		_size = other._size;
		_isize = other._isize;
		_osize = other._osize;

		_layers = new Layer <T> [_size];
		for (size_t i = 0; i < _size; i++)
			_layers[i] = other._layers[i];
		
		init_cache();
	}

	return *this;
}

template <class T>
template <class ... U>
size_t DNN <T> ::fill(size_t i, const Layer <T> &layer, U ... args)
{
	_layers[i] = layer;

	return fill(i + 1, args...);
}

template <class T>
size_t DNN <T> ::fill(size_t i, const Layer <T> &layer)
{
	_layers[i] = layer;

	return _layers[i].get_fan_out();
}

template <class T>
void DNN <T> ::clear()
{
	if (_layers)
		delete[] _layers;
	
	clear_cache();
}

template <class T>
void DNN <T> ::clear_cache()
{
	if (_acache)
		delete[] _acache;
	if (_zcache)
		delete[] _zcache;
	
	if (_Acache)
		delete[] _Acache;
	if (_Zcache)
		delete[] _Zcache;
}

template <class T>
void DNN <T> ::init_cache()
{
	clear_cache();

	_acache = new Vector <T> [_size + 1];
	_zcache = new Vector <T> [_size];

	_Acache = new Matrix <T> [_size + 1];
	_Zcache = new Matrix <T> [_size];
}

// TODO: another file (cpu)
#ifndef __AVR	// Does not support AVR

// Saving and loading
template <class T>
void DNN <T> ::save(const std::string &file)
{
	std::ofstream fout(file);

	fout.write((char *) &_size, sizeof(size_t));
	fout.write((char *) &_isize, sizeof(size_t));
	fout.write((char *) &_osize, sizeof(size_t));

	for (int i = 0; i < _size; i++)
		_layers[i].write(fout);
}

template <class T>
void DNN <T> ::load(const std::string &file)
{
	// Clear the current members
	clear();

	std::ifstream fin(file);

	fin.read((char *) &_size, sizeof(size_t));
	fin.read((char *) &_isize, sizeof(size_t));
	fin.read((char *) &_osize, sizeof(size_t));

	_layers = new Layer <T> [_size];
	for (size_t i = 0; i < _size; i++)
		_layers[i].read(fin);
}

#endif		// Does not support AVR

// Properties
template <class T>
size_t DNN <T> ::size() const
{
	return _size;
}

template <class T>
size_t DNN <T> ::input_size() const
{
	return _isize;
}

template <class T>
size_t DNN <T> ::output_size() const
{
	return _osize;
}

template <class T>
Layer <T> *DNN <T> ::layers()
{
	return _layers;
}

template <class T>
Vector <T> *DNN <T> ::acache() const
{
	return _acache;
}

template <class T>
Vector <T> *DNN <T> ::zcache() const
{
	return _zcache;
}

/*
template <class T>
void DNN <T> ::load_json(const std::string &file)
{
	std::ifstream fin(file);

	nlohmann::json structure;

	fin >> structure;

	auto layers = structure["Layers"];

	// Allocate size information
	_size = layers.size();
	_isize = layers[0]["Neurons"];
	_osize = layers[_size - 1]["Neurons"];

	// Allocate matrices and activations
	_weights = new Matrix <T> [_size - 1];
	_momentum = new Matrix <T> [_size - 1];
	_layers = new Layer <T> [_size];

	// Allocate caches
	_a = std::vector <Vector <T>> (_size);
	_z = std::vector <Vector <T>> (_size - 1);

	std::vector <size_t> sizes;
	for (size_t i = 0; i < _size; i++) {
		auto layer = layers[i];

		auto activation = layer["Activation"];
		size_t neurons = layer["Neurons"];

		std::vector <T> args;

		for (auto arg : activation["Arguments"])
			args.push_back(arg);

		std::string name = activation["Name"];

		sizes.push_back(layer["Neurons"]);

		_layers[i].second = Activation <T> ::load(name, args);
	}

	for (size_t i = 1; i < _size; i++) {
		_weights[i - 1] = Matrix <T> (sizes[i], sizes[i - 1] + 1, T(0));
		_momentum[i - 1] = Matrix <T> (sizes[i], sizes[i - 1] + 1, T(0));
	}

	_random = default_initializer <T> {};
	_cmp = default_comparator;
} */

// Computation
template <class T>
Vector <T> DNN <T> ::operator()(const Vector <T> &in)
{
	return compute(in);
}

template <class T>
Vector <T> DNN <T> ::compute(const Vector <T> &in)
{
	Vector <T> tmp = in;

	for (size_t i = 0; i < _size; i++)
		tmp = _layers[i].forward_propogate(tmp);

	return tmp;
}

template <class T>
Matrix <T> *DNN <T> ::jacobian(const Vector <T> &in)
{
	if (in.size() != _isize)
		throw bad_io_dimensions();

	Matrix <T> *J = jacobian_kernel(
		_layers,
		_size,
		_osize,
		_acache,
		_zcache,
		in);

	return J;
}

template <class T>
Matrix <T> *DNN <T> ::jacobian_delta(const Vector <T> &in, Vector <T> &delta)
{
	if (in.size() != _isize || delta.size() != _osize)
		throw bad_io_dimensions();

	Matrix <T> *J = jacobian_kernel(
		_layers,
		_size,
		_osize,
		_acache,
		_zcache,
		in,
		delta);

	return J;
}

template <class T>
void DNN <T> ::apply_gradient(Matrix <T> *J)
{
	for (size_t i = 0; i < _size; i++)
		_layers[i].apply_gradient(J[i]);
}

// Miscellaneous
template <class T>
void DNN <T> ::print() const
{
	for (size_t i = 0; i < _size; i++)
		_layers[i].print();
}

}

}

#endif
