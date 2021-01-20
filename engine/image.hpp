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
	Image(png_bytep *, size_t, size_t, size_t rbytes);
};

Image::Image(png_bytep *data, size_t width, size_t height, size_t rbytes) : Tensor <unsigned char> ({width, height, 4})
{
	for (size_t i = 0; i < height; i++)
		memcpy(__array + i * rbytes, data[i], rbytes);
}

// Thrown when the file cannot be accessed (replace with std)
class bad_file {};

// Thrown when the file being read is not in PNG format
class bad_png {};

// Throw more specific errors
Tensor <unsigned char> read_png(const char *img)
{
	FILE *file = fopen(img, "rb");

	if (!file)
		throw bad_file();

	png_byte header[8];

	if (fread(header, 1, 8, file) != 8)
		throw bad_png();

	if (png_sig_cmp(header, 0, 8))
		throw bad_png();

	png_structp png_ptr = png_create_read_struct(
			PNG_LIBPNG_VER_STRING,
			nullptr,
			nullptr,
			nullptr);

	if (!png_ptr)
		throw bad_png();

	png_infop info_ptr = png_create_info_struct(png_ptr);

	if (!info_ptr)
		throw bad_png();

	if (setjmp(png_jmpbuf(png_ptr)))
		throw bad_png();

	png_init_io(png_ptr, file);
	png_set_sig_bytes(png_ptr, 8);
	png_read_info(png_ptr, info_ptr);

	size_t width = png_get_image_width(png_ptr, info_ptr);
	size_t height = png_get_image_height(png_ptr, info_ptr);

	png_byte color = png_get_color_type(png_ptr, info_ptr);

	size_t rbytes = png_get_rowbytes(png_ptr,info_ptr);

	png_bytep *data = new png_bytep[height];
	for (size_t i = 0; i < height; i++)
		data[i] = new png_byte[rbytes];

	png_read_image(png_ptr, data);

	size_t channels = 1;

	switch (color) {
	case PNG_COLOR_TYPE_RGB:
		channels = 3;
	case PNG_COLOR_TYPE_GRAY_ALPHA:
		channels = 2;
	case PNG_COLOR_TYPE_RGB_ALPHA:
		channels = 4;
	}

	// Close the file
	fclose(file);

	std::vector <size_t> dimensions = {
		width,
		height,
		channels
	};

	return Image(data, width, height, rbytes);
}

}

}

#endif
