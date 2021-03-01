#ifndef NET_NODE_H_
#define NET_NODE_H_

// C/C++ headers
#include <iostream>
#include <vector>

// Engine headers
#include <tensor.hpp>
#include <filter.hpp>

namespace zhetapi {

namespace ml {

template <class T = double>
class NetNode {
	struct iforward {
		size_t		__index	= 0;
		NetNode <T> *	__fw	= nullptr;
	};

	// Only the arrays are recreated, the actual 'pipes'
	// are simply copied by address value
	Tensor <T> **		__ins		= nullptr;
	Tensor <T> **		__outs		= nullptr;

	// Copied as is (filters are not recreated)
	Filter <T> *		__filter	= nullptr;
	
	size_t			__nins		= 0;
	size_t			__nouts		= 0;

	size_t			__index		= 0;

	std::vector <iforward>	__forward	= {};

	std::string		__name		= "";

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
	__name = "NetNode" + std::to_string(++id);
}

template <class T>
NetNode <T> ::NetNode(const NetNode &other)
		: __filter(other.__filter),
		__nins(other.__nins),
		__nouts(other.__nouts),
		__index(other.__index),
		__forward(other.__forwrd),
		__name(other.__name)
{
	copy_pipes(other);
}

template <class T>
NetNode <T> ::NetNode(Filter <T> *filter)
		: __filter(filter)
{
	__name = "NetNode" + std::to_string(++id);
}

template <class T>
NetNode <T> &NetNode <T> ::operator=(const NetNode &other)
{
	if (this != &other) {
		// Clear pipes first
		clear_pipes();

		__filter = other.__filter;
		__nins = other.__nins;
		__nouts = other.__nouts;
		__index = other.__index;
		__forward = other.__forwrd;
		__name = other.__name;

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
	__ins = new Tensor <T> *[__nins];
	__outs = new Tensor <T> *[__nouts];
	
	memcpy(__ins, other.__ins, sizeof(Tensor <T> *) * __nins);
	memcpy(__outs, other.__outs, sizeof(Tensor <T> *) * __nouts);
}

template <class T>
void NetNode <T> ::clear_pipes()
{
	for (size_t i = 0; i < __nins; i++) {
		// Ensure that the 'pipe' has not been destroyed from the
		// other side
		if (__ins[i]->good()) {
			__ins[i]->clear();

			delete __ins[i];
		}

	}
	
	for (size_t i = 0; i < __nouts; i++) {
		// Ensure that the 'pipe' has not been destroyed from the
		// other side
		if (__outs[i]->good()) {
			__ins[i]->clear();

			delete __outs[i];
		}

	}

	delete[] __ins;
	delete[] __outs;
}

// Propoerties

// Notational usage: ts = *(nn1[i])
template <class T>
const Tensor <T> &NetNode <T> ::operator*() const
{
	if (__index >= __nouts)
		throw bad_index();

	return *(__outs[__index]);
}

template <class T>
NetNode <T> &NetNode <T> ::operator[](size_t i)
{
	__index = i;

	return *this;
}

// Notational usage: out_nn[i] << in_nn[j]
template <class T>
NetNode <T> &NetNode <T> ::operator<<(NetNode &in)
{
	// Check output size for this
	if (__index + 1 > __nouts) {
		Tensor <T> **tmp = new Tensor <T> *[__index + 1];

		memcpy(__outs, tmp, sizeof(Tensor <T> *) * (__nouts));

		// Fill empty entries with default initialized tensors
		// (exclude the entry we are going to create)
		for (size_t i = __nouts; i < __index; i++)
			tmp[i] = new Tensor <T> ();

		__nouts = __index + 1;

		delete[] __outs;

		__outs = tmp;
	}

	// Checkout input size for in
	if (in.__index + 1 > in.__nins) {
		Tensor <T> **tmp = new Tensor <T> *[in.__index + 1];

		memcpy(in.__ins, tmp, sizeof(Tensor <T> *) * (in.__nins));
		
		// Initialize unused entries
		for (size_t i = in.__nins; i < in.__index; i++)
			tmp[i] = new Tensor <T> ();

		in.__nins = in.__index + 1;

		delete[] in.__ins;

		in.__ins = tmp;
	}

	// Clear former connections
	Tensor <T> **ocon = &(__outs[__index]);
	if (*ocon)
		delete *ocon;

	Tensor <T> **icon = &(in.__ins[in.__index]);
	if (*icon)
		delete *icon;

	// Create the new connection
	Tensor <T> *con = new Tensor <T> ();	// Default initialize

	*icon = *ocon = con;

	// Set this as the forward of in
	in.__forward.push_back({in.__index, this});

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
	return __forward;
}

// Notational usage: nn[i].pass(ts)
template <class T>
void NetNode <T> ::pass(const Tensor <T> &ts) const
{
	if (__index >= __nins)
		throw bad_index();

	// Try to avoid copying here,
	// or achieve this through r-value
	// reference parameter overload
	*(__ins[__index]) = ts;
}

// Starts from index 0
template <class T>
void NetNode <T> ::pass(std::vector <Tensor <T>> &args) const
{
	size_t i = 0;
	while (!args.empty() && i < __nins) {
		*(__ins[i++]) = args[0];

		args.erase(args.begin());
	}
}

// Computational methods
template <class T>
void NetNode <T> ::propogate() const
{
	std::vector <Tensor <T> *> inputs;
	std::vector <Tensor <T> *> outputs;

	for (size_t i = 0; i < __nins; i++)
		inputs.push_back(__ins[i]);
	
	for (size_t i = 0; i < __nouts; i++)
		outputs.push_back(__outs[i]);

	__filter->process(inputs, outputs);
}

// Show the flow of ouput from this node
template <class T>
void NetNode <T> ::trace(size_t tabs) const
{
	std::string tb(tabs, '\t');

	std::cout << __name << std::endl;
	for (iforward ifw : __forward) {
		std::cout << tb << "\t@" << ifw.__index << ": ";

		ifw.__fw->trace(tabs + 1);
	}
}

}

}

#endif
