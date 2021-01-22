#ifndef IMAGE_H_
#define IMAGE_H_

// C++ headers
#include <cstring>
#include <bitset>

// GLFW
#include <glad.h>	// Replace this header
#include <GLFW/glfw3.h>

// PNG library
#include <png.h>

// Engine headers
#include <tensor.hpp>
#include <core/shader.hpp>

namespace zhetapi {

namespace image {

// Image class
class Image : public Tensor <unsigned char> {
public:
	Image(png_bytep *, size_t, size_t, size_t, size_t);

	size_t width() const;
	size_t height() const;
	size_t channels() const;

	const unsigned char *const raw() const;

	unsigned char **row_bytes() const;

	int show() const;
};

// Thrown when the file cannot be accessed (replace with std)
class bad_file {};

// Thrown when the file being read is not in PNG format
class bad_png {};

// Image loading and saving
Image load_png(const char *);
void save_png(Image, const char *);

// GLFW helpers
void image_viewer_input_processor(GLFWwindow *);
void image_viewer_resize_processor(GLFWwindow *, int, int);

}

}

#endif
