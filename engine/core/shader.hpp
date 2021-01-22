#ifndef SHADER_H_
#define SHADER_H_

// C++ headers
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// Remove request to local header
#include <glad.h>

namespace zhetapi {

namespace graphics {

unsigned int create_image_shader();
void check_gl_compile_errors(unsigned int, const std::string &);

}

}

#endif
