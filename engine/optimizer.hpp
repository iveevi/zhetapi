#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// Engine headers
#include <network.hpp>

namespace zhetapi {

namespace ml {

// Optimizer class
template <class T>
class Optimizer {
public:
	virtual Matrix <T> *gradient(NeuralNetwork <T> &) = 0;
};

}

}

#endif
