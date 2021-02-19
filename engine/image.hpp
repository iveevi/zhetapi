#ifndef IMAGE_H_
#define IMAGE_H_

// C++ headers
#include <cstring>
#include <bitset>

#ifndef ZHP_NO_GUI

// GLFW
#include <glad.h>	// Replace this header
#include <GLFW/glfw3.h>

#endif

// PNG library
#include <png.h>

// Engine headers
#include <tensor.hpp>
#include <vector.hpp>

#include <core/shader.hpp>

namespace zhetapi {

namespace image {

// Image class
class Image : public Tensor <unsigned char> {
public:
	// Using declararations
	using byte = unsigned char;
	using pixel = std::pair <size_t, size_t>;

	Image();						// Default
	Image(byte *, size_t, size_t, size_t = 1);		// Contigous array
	Image(byte **, size_t, size_t, size_t);			// List of rows
	Image(png_bytep *, size_t, size_t, size_t, size_t);	// (Pretty much the same as above)

	Vector <size_t> size() const;

	size_t width() const;
	size_t height() const;
	size_t channels() const;
	
	// Pixel value setter
	void set(const pixel &, size_t, byte);
	void set(const pixel &, const Vector <byte> &);

	// Image extractors
	Image channel(size_t) const;
	Image crop(const pixel &, const pixel &) const;

	const unsigned char *const raw() const;

	unsigned char **row_bytes() const;

#ifndef ZHP_NO_GUI

	int show() const;

#endif

	class out_of_bounds {};
	class bad_input_order {};

	// Friends
	template <class T>
	friend class Convolution;
protected:
	bool in_bounds(const pixel &) const;
};

// Thrown when the file cannot be accessed (replace with std)
class bad_file {};

// Thrown when the file being read is not in PNG format
class bad_png {};

// Image loading and saving
Image load_png(const std::string &);
Image load_png(const char *);
void save_png(Image, const char *);

#ifndef ZHP_NO_GUI

// GLFW helpers
void image_viewer_input_processor(GLFWwindow *);
void image_viewer_resize_processor(GLFWwindow *, int, int);

#endif

}

}

#endif
