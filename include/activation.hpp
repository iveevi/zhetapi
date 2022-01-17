#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

// Standard headers
#include <algorithm>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Engine headers
#include "cuda/essentials.cuh"
#include "vector.hpp"

// A is class name, T is the type (template), L is the loader function
#define _zhp_register_activation(A, T, L)			\
	zhetapi::ml::Activation <T>				\
	::_act_loaders_id[typeid(A <T>).name()] = L;		\
	zhetapi::ml::Activation <T>				\
	::_act_loaders_name[#A] = L;

namespace zhetapi {

namespace ml {

template <class T>
class Activation;

// Format of an activation loader
template <class T>
using Loader = Activation <T> *(*)(const std::vector <T> &);

/**
 * @brief Represents an activation in machine learning. Takes a vector of type T
 * as an input and returns a vector of type T.
 *
 * @tparam T the type that the activation performs its operations on.
 */
template <class T>
class Activation {
public:
	// TODO: Replace with a string
	enum activation_type {
		AT_Default,
		AT_Linear,
		AT_ReLU,
		AT_Sigmoid
	};

	__cuda_dual__
	Activation();						// Default constructor

	__cuda_dual__
	explicit Activation(activation_type);				// Type constructor

	__cuda_dual__
	explicit Activation(const std::vector <T> &);			// Argument constructor

	__cuda_dual__
	Activation(activation_type, const std::vector <T> &);		// Type and argument constructor

	virtual Activation *copy() const;

	// Computation
	__cuda_dual__
	virtual Vector <T> compute(const Vector <T> &) const;

	__cuda_dual__
	Vector <T> operator()(const Vector <T> &) const;

	__cuda_dual__
	virtual Activation *derivative() const;

	__cuda_dual__
	int get_activation_type() const;

	template <class U>
	__cuda_dual__
	friend Activation <U> *copy(Activation <U> *);

#ifndef __AVR	// Does not support AVR

	// Saving
	void write_type(std::ofstream &) const;
	void write_args(std::ofstream &) const;

	virtual void write(std::ofstream &) const;

	// Loading
	static Activation <T> *load(std::ifstream &);
	static Activation <T> *load(const std::string &, const std::vector <T> &);

	// Global list of all registered activations
	static std::map <std::string, Loader <T>> _act_loaders_id;
	static std::map <std::string, Loader <T>> _act_loaders_name;

	/**
	 * @brief Exception thrown when trying to read activation data from a
	 * file that is not registered in the activation database.
	 */
	class undefined_loader {}; // TODO: need to derive from std::runtime_error

#endif		// Does not support AVR

protected:
	activation_type	_kind = AT_Default;

	// Arguments, empty by default
	std::vector <T>	_args = {};
};

#ifndef __AVR	// Does not support AVR

template <class T>
std::map <std::string, Loader <T>> Activation <T> ::_act_loaders_id;

template <class T>
std::map <std::string, Loader <T>> Activation <T> ::_act_loaders_name;

#endif		// Does not support AVR

#ifndef ZHP_CUDA

// Constructors
template <class T>
Activation <T> ::Activation() : _kind(AT_Default) {}

template <class T>
Activation <T> ::Activation(activation_type kind) : _kind(kind) {}

// TODO: new file (add a new directory for cpu-generic headers)
#ifndef __AVR	// Does not support AVR

template <class T>
Activation <T> ::Activation(const std::vector <T> &args) : _args(args),
		_kind(AT_Default) {}

template <class T>
Activation <T> ::Activation(activation_type kind, const std::vector <T> &args)
		: _args(args), _kind(kind) {}

#endif		// Does not support AVR

/**
 * @brief Default copy method for activations. Displays warning if this method
 * is not overriden by a subclass.
 */
template <class T>
Activation <T> *Activation <T> ::copy() const
{
	static const char *msg = "Warning (from activation.hpp): using the default copy method.";
	std::cerr << msg << std::endl;
	return new Activation <T> ();
}

// TODO: Reverse compute and operator()
template <class T>
Vector <T> Activation <T> ::operator()(const Vector <T> &x) const
{
	return this->compute(x);
}

template <class T>
Vector <T> Activation <T> ::compute(const Vector <T> &x) const
{
	static const char *msg = "Warning (from activation.hpp): using the default compute method.";
	std::cerr << msg << std::endl;
	return x;
}

#ifndef __AVR	// Does not support AVR

// Saving
template <class T>
void Activation <T> ::write_type(std::ofstream &fout) const
{
	std::string aname = typeid(*this).name();
	size_t len = aname.length();

	fout.write((char *) &len, sizeof(size_t));

	fout << aname;
}

template <class T>
void Activation <T> ::write_args(std::ofstream &fout) const
{
	size_t argc = _args.size();

	fout.write((char *) &argc, sizeof(size_t));

	for (size_t i = 0; i < argc; i++)
		fout.write((char *) &(_args[i]), sizeof(T));
}

template <class T>
void Activation <T> ::write(std::ofstream &fout) const
{
	write_type(fout);
	write_args(fout);
}

// Loading
template <class T>
Activation <T> *Activation <T> ::load(std::ifstream &fin)
{
	size_t len;

	fin.read((char *) &len, sizeof(size_t));

	char *aname = new char[len + 1];

	fin.read(aname, sizeof(char) * (len));

	aname[len] = '\0';

	std::string name = aname;

	size_t argc;

	fin.read((char *) &argc, sizeof(size_t));

	std::vector <T> args;
	for (size_t i = 0; i < argc; i++) {
		T t;
		fin.read((char *) &t, sizeof(T));
		args.push_back(t);
	}

	if (_act_loaders_id.find(name) == _act_loaders_id.end())
		throw undefined_loader();

	Loader <T> loader = _act_loaders_id[name];

	delete[] aname;

	return loader(args);
}

template <class T>
Activation <T> *Activation <T> ::load(const std::string &name, const std::vector <T> &args)
{
	if (_act_loaders_name.find(name) == _act_loaders_name.end())
		throw undefined_loader();

	Loader <T> loader = _act_loaders_name[name];

	return loader(args);
}

#endif		// Does not support AVR

// Derivative
template <class T>
Activation <T> *Activation <T> ::derivative() const
{
	return new Activation();
}

template <class T>
int Activation <T> ::get_activation_type() const
{
	return _kind;
}

#endif

}

}

#endif
