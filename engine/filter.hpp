#ifndef FILTER_H_
#define FILTER_H_

// Engine headers
#include <image.hpp>
#include <matrix.hpp>

namespace zhetapi {

namespace image {

class Filter {
	Matrix <unsigned char>	__filter;
public:
	Filter(const Matrix <unsigned char> &);
};

}

}

#endif