#include <image.hpp>

namespace zhetapi {

namespace image {

// Get hex digit value
static byte hex_to_byte(char c)
{
	if (isdigit(c))
		return c - '0';

	if (isupper(c))
		return (c - 'A') + 10;

	// Assume lower-case letter
	return (c - 'a') + 10;
};

// Color constructors
Color::Color() {}

Color::Color(const char *str) : Color(std::string(str)) {}

Color::Color(const std::string &hexs)
{
	static const size_t LENGTH = 7;

	if (hexs.length() != LENGTH)
		throw bad_hex_string();

	r = 16 * hex_to_byte(hexs[1]) + hex_to_byte(hexs[2]);
	g = 16 * hex_to_byte(hexs[3]) + hex_to_byte(hexs[4]);
	b = 16 * hex_to_byte(hexs[5]) + hex_to_byte(hexs[6]);
};

Color::Color(byte xr, byte xg, byte xb) : r(xr), g(xg), b(xb) {}

// Color properties

// Return rgb in decimal form (each digit is a byte)
uint32_t Color::value() const
{
	uint32_t ur = r;
	uint32_t ug = g;
	uint32_t ub = b;

	return (((ur << 8) + ug) << 8) + ub;
}

// Standard colors
const Color	RED	= "#FF0000";
const Color	GREEN	= "#00FF00";
const Color	BLUE	= "#0000FF";
const Color	YELLOW	= "#FFFF00";
const Color	ORANGE	= "#FFA500";
const Color	CYAN	= "#00FFFF";
const Color	WHITE	= "#FFFFFF";
const Color	BLACK	= "#000000";
const Color	GREY	= "#808080";

// Gradient constructor
Gradient::Gradient(const Color &A, const Color &B,
		long double start, long double end)
		: _base(A), _start(start), _end(end)
{
	_dr = B.r - A.r;
	_dg = B.g - A.g;
	_db = B.b - A.b;
}

Gradient::Gradient(const std::string &hexs_a, const std::string &hexs_b,
		long double start, long double end)
		: _base(hexs_a), _start(start), _end(end)
{
	Color B = hexs_b;

	_dr = B.r - _base.r;
	_dg = B.g - _base.g;
	_db = B.b - _base.b;
}

// Gradient getter
Color Gradient::get(long double x)
{
	long double k = (x - _start)/(_start - _end);

	return Color {
		(byte) (_base.r + _dr * k),
		(byte) (_base.g + _dg * k),
		(byte) (_base.b + _db * k)
	};
}

// Image constructors
Image::Image() : Tensor <unsigned char> () {}

Image::Image(size_t width, size_t height, size_t channels, byte def)
		: Tensor <unsigned char> ({width, height, channels}, def) {}

Image::Image(size_t width, size_t height, size_t channels, const Color &color)
		: Tensor <unsigned char> ({width, height, channels})
{
	for (size_t r = 0; r < width; r++) {
		for (size_t c = 0; c < height; c++)
			set(pixel {r, c}, color);
	}
}

Image::Image(size_t width, size_t height, size_t channels, const std::string &hexs)
		: Tensor <unsigned char> ({width, height, channels})
{
	Color color = hexs;
	for (size_t r = 0; r < width; r++) {
		for (size_t c = 0; c < height; c++)
			set(pixel {r, c}, color);
	}
}

// Reinterpret constructor (row-contingious data)
Image::Image(byte *data, size_t width, size_t height, size_t channels)
		: Tensor <unsigned char> ({width, height, channels})
{
	size_t nbytes = width * height * channels;
	memcpy(_array, data, nbytes);
}

Image::Image(byte **data, size_t width, size_t height, size_t channels)
		: Tensor <unsigned char> ({width, height, channels})
{
	size_t rbytes = width * channels;
	for (size_t i = 0; i < height; i++)
		memcpy(_array + i * rbytes, data[i], rbytes);
}

// TODO: Resolve the similarity between this and the above constructor
Image::Image(png_bytep *data, size_t width, size_t height, size_t channels, size_t rbytes)
		: Tensor <unsigned char> ({width, height, channels})
{
	for (size_t i = 0; i < height; i++)
		memcpy(_array + i * rbytes, data[i], rbytes);
}

Image::Image(const Vector <double> &values, size_t width, size_t height)
		: Tensor <unsigned char> ({width, height, 1})
{
	for (size_t i = 0; i < this->_size; i++)
		_array[i] = (unsigned char) values[i];
}

Vector <size_t> Image::size() const
{
	return {
		_dim[0],
		_dim[1]
	};
}

size_t Image::width() const
{
	return _dim[0];
}

size_t Image::height() const
{
	return _dim[1];
}

size_t Image::channels() const
{
	return _dim[2];
}

sf::Image Image::sfml_image() const
{
	sf::Image image;

	// TODO: assert RGBA and/or fill
	image.create(_dim[0], _dim[1], _array);

	return image;
}

sf::Texture Image::sfml_texture() const
{
	sf::Texture texture;
	texture.loadFromImage(sfml_image());
	return texture;
}

void Image::set(const pixel &px, const Color &c)
{
	size_t index = _dim[2] * (px.first * _dim[1] + px.second);

	_array[index] = c.r;
	_array[index + 1] = c.g;
	_array[index + 2] = c.b;
}

void Image::set(const pixel &px, size_t channel, byte bt)
{
	size_t index = _dim[2] * (px.first * _dim[1] + px.second);

	_array[index + channel] = bt;
}

void Image::set(const pixel &px, const Vector <byte> &bytes)
{
	size_t index = _dim[2] * (px.first * _dim[1] + px.second);

	size_t nbytes = bytes.size();
	for (size_t i = 0; i < nbytes && i + index < _size; i++)
		_array[i + index] = bytes[i];
}

void Image::set_hex(const pixel &px, size_t hexc)
{
	byte r = (byte) ((hexc & 0xFF0000) >> 16);
	byte g = (byte) ((hexc & 0x00FF00) >> 8);
	byte b = (byte) (hexc & 0x0000FF);

	set(px, Color(r, g, b));
}

void Image::set_hex(const pixel &px, const std::string &hexs)
{
	set(px, Color(hexs));
}

// Pixel value getter (same value as Color::value)
uint32_t Image::color(const pixel &px) const
{
	size_t index = _dim[2] * (px.first * _dim[1] + px.second);

	uint32_t ur = _array[index];
	uint32_t ug = _array[index + 1];
	uint32_t ub = _array[index + 2];
	
	return (((ur << 8) + ug) << 8) + ub;
}

// Extract a single channel
Image Image::channel(size_t channels) const
{
	size_t len = _dim[0] * _dim[1];

	byte *data = new byte[len];
	for (size_t i = 0; i < len; i++)
		data[i] = _array[i * _dim[2] + channels];

	Image out(data, _dim[0], _dim[1]);

	// Free memory
	delete[] data;

	return out;
}

// Cropping images (from top-right to bottom-left)
Image Image::crop(const pixel &tr, const pixel &bl) const
{
	if (bl <= tr)
		throw bad_input_order();

	if (!in_bounds(tr) || !in_bounds(bl))
		throw out_of_bounds();

	size_t s_index = tr.first * _dim[1] + tr.second;
	// size_t e_index = bl.first * _dim[1] + bl.second;

	size_t n_width = bl.first - tr.first + 1;
	size_t n_height = bl.second - tr.second + 1;

	byte **rows = new byte *[n_height];
	for (size_t i = 0; i < n_height; i++) {
		size_t k = _dim[2] * (s_index + i * _dim[1]);
		rows[i] = &(_array[k]);
	}

	Image out(rows, n_width, n_height, _dim[2]);

	// Free the memory
	delete[] rows;

	return out;
}

const unsigned char *const Image::raw() const
{
	return _array;
}

unsigned char **Image::row_bytes() const
{
	unsigned char **rows = new unsigned char *[_dim[0]];

	size_t stride = _dim[1] * _dim[2];
	for (size_t i = 0; i < _dim[0]; i++)
		rows[i] = &(_array[stride * i]);

	return rows;
}

bool Image::in_bounds(const pixel &px) const
{
	return (px.first >= 0 && px.first < _dim[0])
		&& (px.second >= 0 && px.second < _dim[1]);
}

#ifndef ZHP_NO_GUI

int Image::show() const
{
	sf::ContextSettings glsettings;
	glsettings.antialiasingLevel = 2;

	// TODO: add optional names to images later
	sf::RenderWindow window {
		sf::VideoMode {
			static_cast <unsigned int> (_dim[0]),
			static_cast <unsigned int> (_dim[1])
		},
		"Image show",
		sf::Style::Titlebar | sf::Style::Close,
		glsettings
	};

	// TODO: this is RGBA only
	sf::Image image;
	image.create(_dim[0], _dim[1], _array);

	sf::Texture texture;
	texture.loadFromImage(image);

	sf::Sprite sprite;
	sprite.setTexture(texture);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}

		window.clear(sf::Color::Black);
		window.draw(sprite);
		window.display();
	}

	return 0;
}

#endif

/*
 * PNG Parsing.
 *
 * TODO: Throw more specific exceptions.
 */
Image load_png(FILE *file)
{
	if (!file)
		throw bad_file();

	png_byte header[8];

	if (fread(header, 1, 8, file) != 8) {
		fclose(file);

		throw bad_png();
	}

	if (png_sig_cmp(header, 0, 8)) {
		fclose(file);

		throw bad_png();
	}

	png_structp png_ptr = png_create_read_struct(
			PNG_LIBPNG_VER_STRING,
			nullptr,
			nullptr,
			nullptr);

	if (!png_ptr) {
		fclose(file);

		throw bad_png();
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);

	if (!info_ptr) {
		fclose(file);

		throw bad_png();
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		fclose(file);

		throw bad_png();
	}

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
		break;
	case PNG_COLOR_TYPE_GRAY_ALPHA:
		channels = 2;
		break;
	case PNG_COLOR_TYPE_RGB_ALPHA:
		channels = 4;
		break;
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

Image load_png(const char *impath)
{
	FILE *file = fopen(impath, "rb");

	return load_png(file);
}

Image load_png(const std::string &impath)
{
	return load_png(impath.c_str());
}

void save_png(const Image &img, const char *path)
{
	FILE *file = fopen(path, "wb");

	if(!file)
		abort();

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png)
		abort();

	png_infop info = png_create_info_struct(png);
	if (!info)
		abort();

	if (setjmp(png_jmpbuf(png))) abort();

	png_init_io(png, file);

	// Output is 8bit depth, RGBA format.
	size_t color = -1;

	// TODO: Needs to be changed (store in Image)
	switch (img.channels()) {
	case 4:
		color = PNG_COLOR_TYPE_RGBA;
		break;
	default:
		color = PNG_COLOR_TYPE_GRAY;
		break;
	}

	png_set_IHDR(
		png,
		info,
		img.width(),
		img.height(),
		8,
		color,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT
	);

	png_write_info(png, info);

	unsigned char **data = img.row_bytes();

	png_write_image(png, data);
	png_write_end(png, NULL);

	delete[] data;

	fclose(file);

	png_destroy_write_struct(&png, &info);
}

}

// Literal operators
image::Image operator""_png(const char *str, size_t len)
{
	std::string file(str, len);

	size_t i = file.find_last_of(".");

	if (i == std::string::npos)
		file += ".png";

	return image::load_png(file);
}

}
