#ifndef LAYERS_H_
#define LAYERS_H_

// Engine headers
#include <matrix.hpp>
#include <activation.hpp>

namespace zhetapi {

namespace ml {

template <class T>
struct Layer {
	Matrix <T> mat;
	Activation <T> act;
	std::function <T ()> initializer;
};

}

}

#endif