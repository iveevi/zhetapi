#ifndef GNN_H_
#define GNN_H_

// C/C++ headers
#include <vector>

// Engine headers
#include <netnode.hpp>

namespace zhetapi {

namespace ml {

/*
 * General neural network (GNN):
 *
 * Represents a neural network whose structure is theoretically unlimited, i.e.
 * a neural network with various kinds of connections (skip connections, etc.)
 * and different types of layers (deep layer, convolutional layer, recurrent layer).
 */
template <class T = double>
class GNN {
	std::vector <NetNode <T>>	__ins = {};
public:
	GNN(const std::vector <NetNode <T>> &);
};

template <class T>
GNN <T> ::GNN(const std::vector <NetNode <T>> &ins)
		: __ins(ins) {}

}

}

#endif
