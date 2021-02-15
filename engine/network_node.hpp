#ifndef NETWORK_NODE_H_
#define NETWORK_NODE_H_

// C/C++ headers
#include <vector>

// Engine headers
#include <tensor.hpp>
#include <filter.hpp>

namespace zhetapi {

namespace ml {

template <class T = double>
class NetworkNode {
	Tensor <T> **	__ins	= nullptr;
	Tensor <T> **	__outs	= nullptr;

	Filter <T> *	__filter = nullptr;
	
	size_t		__nins	= 0;
	size_t		__nouts	= 0;

	size_t		__index	= 0;
public:
	NetworkNode();
	NetworkNode(Filter *);

	~NetworkNode();

	NetworkNode &operator[](size_t);

	NetworkNode &operator<<(NetworkNode &);
};

template <class T>
NetworkNode <T> ::NetworkNode() {}

template <class T>
NetworkNode <T> ::NetworkNode(Filter *filter)
		: __filter(filter) {}

template <class T>
NetworkNode <T> ::~NetworkNode()
{
	for (size_t i = 0; i < __nins; i++) {
		// Ensure that the 'pipe' has not been destroyed from the
		// other side
		if (__ins[i])
			delete __ins[i];

	}
	
	for (size_t i = 0; i < __nouts; i++) {
		// Ensure that the 'pipe' has not been destroyed from the
		// other side
		if (__outs[i])
			delete __outs[i];

	}

	delete[] __ins;
	delete[] __outs;
}

template <class T>
NetworkNode &NetworkNode <T> ::operator[](size_t i)
{
	__index = i;

	return *this;
}

template <class T>
NetworkNode <T> &NetworkNode <T> ::operator<<(NetworkNode &out)
{
	// Check input size for this
	if (__index + 1 > __nins) {
		Tensor <T> **tmp = new Tensor <T> *[__index + 1];

		memcpy(__ins, tmp, sizeof(Tensor <T> *) * (__nins));

		__nins = __index + 1;
	}

	// Checkout output size for out
	if (out.__index + 1 > out.__nouts) {
		Tensor <T> **tmp = new Tensor <T> *[out.__index + 1];

		memcpy(out.__outs, tmp, sizeof(Tensor <T> *) * (out.__nouts));

		out.__nouts = out.__index + 1;
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

	// Allow next object to 'pipe' with this one
	return *this
}

}

}

#endif
