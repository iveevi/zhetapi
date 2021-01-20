#ifndef IMAGE_H_
#define IMAGE_H_

// C++ headers
#include <cstring>
#include <bitset>

// PNG library
#include <png.h>

// Engine headers
#include <tensor.hpp>

namespace zhetapi {

namespace image {

// Image class
class Image : public Tensor <unsigned char> {
public:
	Image(png_bytep *, size_t, size_t, size_t, size_t);
};

// Thrown when the file cannot be accessed (replace with std)
class bad_file {};

// Thrown when the file being read is not in PNG format
class bad_png {};

// Image parsing
Image read_png(const char *);

}

}

#endif
