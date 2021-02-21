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
	Tensor <T> **		__ins		= nullptr;
	Tensor <T> **		__outs		= nullptr;

	Filter <T> *		__filter	= nullptr;
	
	size_t			__nins		= 0;
	size_t			__nouts		= 0;

	size_t			__index		= 0;

	std::vector <NetNode *>	__forward	= {};

	std::string		__name		= "";
public:
	NetNode();
	NetNode(Filter <T> *);

	~NetNode();

	// Essential overloaded operators
	const Tensor <T> &operator*() const;	// Returns indexed OUTPUT

	NetNode &operator[](size_t);		// Sets index
	NetNode &operator<<(NetNode &);		// Creates a connection

	// Setters and getters
	void pass(const Tensor <T> &) const;

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
NetNode <T> ::NetNode(Filter <T> *filter)
		: __filter(filter)
{
	__name = "NetNode" + std::to_string(++id);
}

template <class T>
NetNode <T> ::~NetNode()
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

// Notational usage: nn1[i] << nn2[j]
template <class T>
NetNode <T> &NetNode <T> ::operator<<(NetNode &out)
{
	// Check input size for this
	if (__index + 1 > __nins) {
		Tensor <T> **tmp = new Tensor <T> *[__index + 1];

		memcpy(__ins, tmp, sizeof(Tensor <T> *) * (__nins));

		__nins = __index + 1;

		delete[] __ins;

		__ins = tmp;
	}

	// Checkout output size for out
	if (out.__index + 1 > out.__nouts) {
		Tensor <T> **tmp = new Tensor <T> *[out.__index + 1];

		memcpy(out.__outs, tmp, sizeof(Tensor <T> *) * (out.__nouts));

		out.__nouts = out.__index + 1;

		delete[] out.__outs;

		out.__outs = tmp;
	}

	// Clear former connections
	Tensor <T> **icon = &(__ins[__index]);
	if (*icon)
		delete *icon;

	Tensor <T> **ocon = &(out.__outs[out.__index]);
	if (*ocon)
		delete *ocon;

	// Create the new connection
	Tensor <T> *con = new Tensor <T> ();	// Default initialize

	*icon = *ocon = con;

	// Set out as the forward of this
	__forward.push_back(&out);

	// Allow next object to 'pipe' with this one
	return *this;
}

// Setters and getters

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
	std::cout << std::string(tabs, '\t') << __name << std::endl;
	for (NetNode *nn : __forward)
		nn->trace(tabs + 1);
}

}

}

#endif
