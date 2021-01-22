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

#ifndef ZHP_NO_GUI

int Image::show() const
{
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(__dim[0], __dim[1], "Zhetapi Image Viewer", NULL, NULL);

	if (!window) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, image_viewer_resize_processor);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	unsigned int shader = graphics::create_image_shader();

	float vertices[] = {
		// positions          // colors           // texture coords
		1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 0.0f, // top right
		1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 1.0f, // bottom right
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 1.0f, // bottom left
		-1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 0.0f  // top left 
	};
	unsigned int indices[] = {  
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};

	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);


	// load and create a texture 
	// -------------------------
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// load image, create texture and generate mipmaps
	int width, height, nrChannels;
	
	const unsigned char *data = __array;
	if (data) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, __dim[0], __dim[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	} else {
		std::cout << "Failed to load texture" << std::endl;
	}

	while (!glfwWindowShouldClose(window)) {
		image_viewer_input_processor(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glBindTexture(GL_TEXTURE_2D, texture);

		glUseProgram(shader);

		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	glfwTerminate();

	return 0;
}

#endif

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

	unsigned char **data = img.row_bytes();

	png_write_image(png, data);
	png_write_end(png, NULL);

	delete[] data;

	fclose(file);

	png_destroy_write_struct(&png, &info);
}

#ifndef ZHP_NO_GUI

// GLFW helpers
void image_viewer_input_processor(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void image_viewer_resize_processor(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

#endif

}

}