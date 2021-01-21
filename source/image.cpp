#include <image.hpp>

namespace zhetapi {

namespace image {

Image::Image(png_bytep *data, size_t width, size_t height, size_t channels, size_t rbytes)
		: Tensor <unsigned char> ({width, height, channels})
{
	for (size_t i = 0; i < height; i++)
		memcpy(__array + i * rbytes, data[i], rbytes);
}

size_t Image::width() const
{
	return __dim[0];
}

size_t Image::height() const
{
	return __dim[1];
}

size_t Image::channels() const
{
	return __dim[2];
}

const unsigned char *const Image::raw() const
{
	return __array;
}

unsigned char **Image::row_bytes() const
{
	unsigned char **rows = new unsigned char *[__dim[0]];

	size_t stride = __dim[1] * __dim[2];
	for (size_t i = 0; i < __dim[0]; i++)
		rows[i] = &(__array[stride * i]);

	return rows;
}

/*
 * PNG Parsing.
 *
 * TODO: Throw more specific exceptions.
 */
Image load_png(const char *impath)
{
	FILE *file = fopen(impath, "rb");

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

	return Image(data, width, height, channels, rbytes);
}

void save_png(Image img, const char *path)
{
	FILE *fp = fopen(path, "wb");

	if(!fp)
		abort();

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png)
		abort();

	png_infop info = png_create_info_struct(png);
	if (!info)
		abort();

	if (setjmp(png_jmpbuf(png))) abort();

	png_init_io(png, fp);

	// Output is 8bit depth, RGBA format.
	png_set_IHDR(
		png,
		info,
		img.width(),
		img.height(),
		8,
		PNG_COLOR_TYPE_RGBA,	// Needs to be changed (store in Image)
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT
	);

	png_write_info(png, info);

	png_write_image(png, img.row_bytes());
	png_write_end(png, NULL);

	fclose(fp);

	png_destroy_write_struct(&png, &info);
}

}

}
