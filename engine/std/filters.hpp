#ifndef STD_FILTERS_H_
#define STD_FILTERS_H_

// Engine headrers
#include <filter.hpp>
#include <matrix.hpp>

namespace zhetapi {

namespace ml {

template <class T>
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
	size_t		__dim;
public:
	Convolution(const Matrix <T> &filter) : __filter(filter), 
			__dim(filter.get_rows()) {}

	// Assume equal padding for now
	Image process(const Image &in) {
		Image out = in;

		size_t w = in.width();
		size_t h = in.height();

		size_t n = (__dim - 1)/2;
		for (size_t i = 0; i < w; i++) {
			for (size_t j = 0; i < h; j++) {
				unsigned char t = 0;

				for (size_t k = 0; k < __dim; k++) {
					if (i + k - n < 0)
						continue;
					
					if (j + k - n < 0) {

					}
				}
			}
		}

		return out;
	}
};

}

}

#endif
