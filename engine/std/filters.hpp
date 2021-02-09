#ifndef STD_FILTERS_H_
#define STD_FILTERS_H_

// Engine headrers
#include <filter.hpp>

namespace zhetapi {

namespace ml {

class FeedForward : public Filter {
	Matrix <T>	__weight;
	Activation <T> *__act;
	Activation <T> *__dact;
public:
};

}

namespace image {

// Assumes that the input tensor is an image
template <class T>
class Convolution : public Filter {
	Matrix <T>	__filter;
public:
	Convolution(const Matrix <T> &filter) : __filter(filter) {}

	// Assume equal padding for now
	Image process(const Image &in) {
		Image out = in;

		return out;
	}
};

}

}

#endif
