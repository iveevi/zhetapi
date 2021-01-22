#include <glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

// #include "shader.hpp"
#include <core/shader.hpp>

#include <image.hpp>

using namespace std;
using namespace zhetapi;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main()
{
	image::Image img = image::load_png("zhetapi-logo.png");

	img.show();
}
