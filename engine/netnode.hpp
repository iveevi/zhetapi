#ifndef NET_NODE_H_
#define NET_NODE_H_

// C/C++ headers
#include <iostream>
#include <vector>

// Engine headers
#include "tensor.hpp"
#include "filter.hpp"

namespace zhetapi {

namespace ml {

template <class T = double>
class NetNode {
	struct iforward {
		size_t		_index	= 0;
		NetNode <T> *	_fw	= nullptr;
	};

	// Only the arrays are recreated, the actual 'pipes'
	// are simply copied by address value
	Tensor <T> **		_ins		= nullptr;
	Tensor <T> **		_outs		= nullptr;

	// Copied as is (filters are not recreated)
	Filter <T> *		_filter	= nullptr;
	
	size_t			_nins		= 0;
	size_t			_nouts		= 0;

	size_t			_index		= 0;

	std::vector <iforward>	_forward	= {};

	std::string		_name		= "";

	// Dealing with pipes
	void copy_pipes(const NetNode &);
	void clear_pipes();

	// Get forward nodes
	const std::vector <iforward> &forward() const;
public:
	NetNode();
	NetNode(const NetNode &);
	NetNode(Filter <T> *);

	NetNode &operator=(const NetNode &);

	~NetNode();

	// Essential overloaded operators
	const Tensor <T> &operator*() const;	// Returns indexed OUTPUT

	NetNode &operator[](size_t);		// Sets index
	NetNode &operator<<(NetNode &);		// Creates a connection
	NetNode &operator>>(NetNode &);		// Creates a connection

	// Setters and getters
	void pass(const Tensor <T> &) const;

	void pass(std::vector <Tensor <T>> &) const;

	// Computational methods
	void propogate() const;

	// Debugging
	void trace(size_t = 0) const;

	// Identifier counter
	static size_t id;

	// Exception classes
	class bad_index {};
};

// Initializing static variables
template <class T>
size_t NetNode <T> ::id = 0;

template <class T>
NetNode <T> ::NetNode()
{
	_name = "NetNode" + std::to_string(++id);
}

template <class T>
NetNode <T> ::NetNode(const NetNode &other)
		: _filter(other._filter),
		_nins(other._nins),
		_nouts(other._nouts),
		_index(other._index),
		_forward(other._forwrd),
		_name(other._name)
{
	copy_pipes(other);
}

template <class T>
NetNode <T> ::NetNode(Filter <T> *filter)
		: _filter(filter)
{
	_name = "NetNode" + std::to_string(++id);
}

template <class T>
NetNode <T> &NetNode <T> ::operator=(const NetNode &other)
{
	if (this != &other) {
		// Clear pipes first
		clear_pipes();

		_filter = other._filter;
		_nins = other._nins;
		_nouts = other._nouts;
		_index = other._index;
		_forward = other._forwrd;
		_name = other._name;

		copy_pipes(other);
	}

	return *this;
}

template <class T>
NetNode <T> ::~NetNode()
{
	clear_pipes();
}

template <class T>
void NetNode <T> ::copy_pipes(const NetNode &other)
{
	_ins = new Tensor <T> *[_nins];
	_outs = new Tensor <T> *[_nouts];
	
	memcpy(_ins, other._ins, sizeof(Tensor <T> *) * _nins);
	memcpy(_outs, other._outs, sizeof(Tensor <T> *) * _nouts);
}

template <class T>
void NetNode <T> ::clear_pipes()
{
	for (size_t i = 0; i < _nins; i++) {
		// Ensure that the 'pipe' has not been destroyed from the
		// other side
		if (_ins[i]->good()) {
			_ins[i]->clear();

			delete _ins[i];
		}

	}
	
	for (size_t i = 0; i < _nouts; i++) {
		// Ensure that the 'pipe' has not been destroyed from the
		// other side
		if (_outs[i]->good()) {
			_ins[i]->clear();

			delete _outs[i];
		}

	}

	delete[] _ins;
	delete[] _outs;
}

// Propoerties

// Notational usage: ts = *(nn1[i])
template <class T>
const Tensor <T> &NetNode <T> ::operator*() const
{
	if (_index >= _nouts)
		throw bad_index();

	return *(_outs[_index]);
}

template <class T>
NetNode <T> &NetNode <T> ::operator[](size_t i)
{
	_index = i;

	return *this;
}

// Notational usage: out_nn[i] << in_nn[j]
template <class T>
NetNode <T> &NetNode <T> ::operator<<(NetNode &in)
{
	// Check output size for this
	if (_index + 1 > _nouts) {
		Tensor <T> **tmp = new Tensor <T> *[_index + 1];

		memcpy(_outs, tmp, sizeof(Tensor <T> *) * (_nouts));

		// Fill empty entries with default initialized tensors
		// (exclude the entry we are going to create)
		for (size_t i = _nouts; i < _index; i++)
			tmp[i] = new Tensor <T> ();

		_nouts = _index + 1;

		delete[] _outs;

		_outs = tmp;
	}

	// Checkout input size for in
	if (in._index + 1 > in._nins) {
		Tensor <T> **tmp = new Tensor <T> *[in._index + 1];

		memcpy(in._ins, tmp, sizeof(Tensor <T> *) * (in._nins));
		
		// Initialize unused entries
		for (size_t i = in._nins; i < in._index; i++)
			tmp[i] = new Tensor <T> ();

		in._nins = in._index + 1;

		delete[] in._ins;

		in._ins = tmp;
	}

	// Clear former connections
	Tensor <T> **ocon = &(_outs[_index]);
	if (*ocon)
		delete *ocon;

	Tensor <T> **icon = &(in._ins[in._index]);
	if (*icon)
		delete *icon;

	// Create the new connection
	Tensor <T> *con = new Tensor <T> ();	// Default initialize

	*icon = *ocon = con;

	// Set this as the forward of in
	in._forward.push_back({in._index, this});

	// Allow next object to 'pipe' with this one
	return *this;
}

// Notational usage: in_nn[i] >> out_nn[j]
template <class T>
NetNode <T> &NetNode <T> ::operator>>(NetNode &out)
{
	out << *this;

	return *this;
}

// Setters and getters
template <class T>
const std::vector <typename NetNode <T> ::iforward>
		&NetNode <T> ::forward() const
{
	return _forward;
}

// Notational usage: nn[i].pass(ts)
template <class T>
void NetNode <T> ::pass(const Tensor <T> &ts) const
{
	if (_index >= _nins)
		throw bad_index();

	// Try to avoid copying here,
	// or achieve this through r-value
	// reference parameter overload
	*(_ins[_index]) = ts;
}

// Starts from index 0
template <class T>
void NetNode <T> ::pass(std::vector <Tensor <T>> &args) const
{
	size_t i = 0;
	while (!args.empty() && i < _nins) {
		*(_ins[i++]) = args[0];

		args.erase(args.begin());
	}
}

// Computational methods
template <class T>
void NetNode <T> ::propogate() const
{
	std::vector <Tensor <T> *> inputs;
	std::vector <Tensor <T> *> outputs;

	for (size_t i = 0; i < _nins; i++)
		inputs.push_back(_ins[i]);
	
	for (size_t i = 0; i < _nouts; i++)
		outputs.push_back(_outs[i]);

	_filter->process(inputs, outputs);
}

// Show the flow of ouput from this node
template <class T>
void NetNode <T> ::trace(size_t tabs) const
{
	std::string tb(tabs, '\t');

	std::cout << _name << std::endl;
	for (iforward ifw : _forward) {
		std::cout << tb << "\t@" << ifw._index << ": ";

		ifw._fw->trace(tabs + 1);
	}
}

}

}

#endif
