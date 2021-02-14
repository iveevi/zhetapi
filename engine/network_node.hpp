#ifndef NETWORK_NODE_H_
#define NETWORK_NODE_H_

// C/C++ headers
#include <vector>

// Engine headers
#include <tensor.hpp>

namespace zhetapi {

namespace ml {

template <class T = double>
class NetworkNode {
	Tensor <T> **	__ins	= nullptr;
	Tensor <T> **	__outs	= nullptr;

	size_t		__nins	= 0;
	size_t		__nouts	= 0;

	size_t		__index	= 0;
public:
	NetworkNode();

	NetworkNode &operator[](size_t);

	NetworkNode &operator<<(NetworkNode &);
};

template <class T>
NetworkNode <T> ::NetworkNode() {}

template <class T>
NetworkNode <T> &NetworkNode <T> ::operator<<(NetworkNode &out)
{
	// Check input size for this
	if (__index + 1 > __nins) {
		Tensor <T> **tmp = new Tensor <T> *[__index + 1];

		memcpy(__ins, tmp, sizeof(Tensor <T> *) * (__nins));

		__nins = __index + 1;
	}
}

}

}

#endif