#ifndef IMAGE_H_
#define IMAGE_H_

// C++ headers
#include <cstring>
#include <bitset>

#ifndef ZHP_NO_GUI

#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

// #include "core/shader.hpp"

#endif

// PNG library
#include <png.h>

// Engine headers
#include "tensor.hpp"
#include "vector.hpp"

namespace zhetapi {

namespace image {

// Global type aliases
using byte = unsigned char;

// Global exceptions
class bad_hex_string {};

// Color structure
//
// TODO: Derive Color from Vector
struct Color {
	byte	r	= 0;
	byte	g	= 0;
	byte	b	= 0;

	Color();
	Color(const char *);			// Hex constructor
	Color(const std::string &);		// Hex constructor
	Color(byte = 0, byte = 0, byte = 0);	// Value constructor

	uint32_t value() const;
};

// Standard colors
extern const Color RED;
extern const Color GREEN;
extern const Color BLUE;
extern const Color YELLOW;
extern const Color ORANGE;
extern const Color CYAN;
extern const Color WHITE;
extern const Color BLACK;
extern const Color GREY;

/*
 * Gradient:
 *
 * A parametrized gradient class, from color A to B, and operating on a range a
 * to b. A value c in the range [a, b] will equate to a color appropriately
 * in between A and B.
 *
 * This class can essentially be thought of as a slider from color A to color B
 * (with the slider value ranging from a to b).
 *
 * The reason we do not restrict a, b = 0, 1 is to allow for more meaningful
 * values. For example, if the gradient is intended to represent heat, the
 * Celcius measurements in [0, 100] are more meaningful to use than are the
 * values in [0, 1].
 */

class Gradient {
	Color		_base;

	long double	_dr	= 0;
	long double	_dg	= 0;
	long double	_db	= 0;

	long double	_start	= 0;
	long double	_end	= 0;
public:
	Gradient(const Color &, const Color &,
			long double = 0, long double = 1);
	Gradient(const std::string &, const std::string &,
			long double = 0, long double = 1);

	Color get(long double);
};

// Image class
class Image : public Tensor <unsigned char> {
public:
	// Using declararations
	using pixel = std::pair <size_t, size_t>;

	Image();						// Default
	Image(size_t, size_t, size_t, byte = 0);		// Value
	Image(size_t, size_t, size_t, const Color &);		// Color
	Image(size_t, size_t, size_t, const std::string &);	// Color
	Image(byte *, size_t, size_t, size_t = 1);		// Contigous array
	Image(byte **, size_t, size_t, size_t);			// List of rows
	Image(png_bytep *, size_t, size_t, size_t, size_t);	// (Pretty much the same as above)

	Image(const Vector <double> &, size_t, size_t);		// Grayscale from vector

	Vector <size_t> size() const;

	size_t width() const;
	size_t height() const;
	size_t channels() const;

	// For SFML
	sf::Image sfml_image() const;
	sf::Texture sfml_texture() const;

	// Pixel value setter
	void set(const pixel &, const Color &);			// Color
	void set(const pixel &, size_t, byte);
	void set(const pixel &, const Vector <byte> &);

	void set_hex(const pixel &, size_t);
	void set_hex(const pixel &, const std::string &);

	// Pixel value getter
	uint32_t color(const pixel &) const;

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
Image load_png(std::ifstream &);

Image load_png(const char *);
Image load_png(const std::string &);

void save_png(const Image &, const char *);

}

// Literal operators
image::Image operator""_png(const char *, size_t);

}

#endif
