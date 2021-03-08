#include <core/shader.hpp>

namespace zhetapi {

namespace graphics {

unsigned int create_image_shader()
{
	static const char *vertex_shader_code =
	"#version 330 core"
	"\nlayout (location = 0) in vec3 aPos;"
	"\nlayout (location = 1) in vec3 aColor;"
	"\nlayout (location = 2) in vec2 aTexCoord;"
	"\nout vec3 ourColor;"
	"\nout vec2 TexCoord;"
	"\nvoid main()"
	"\n{"
	"\n	gl_Position = vec4(aPos, 1.0);"
	"\n	ourColor = aColor;"
	"\n	TexCoord = vec2(aTexCoord.x, aTexCoord.y);"
	"\n}";

	static const char *fragment_shader_code =
	"#version 330 core"
	"\nout vec4 FragColor;"
	"\nin vec3 ourColor;"
	"\nin vec2 TexCoord;"
	"\nuniform sampler2D texture1;"
	"\nvoid main()"
	"\n{"
	"\n	FragColor = texture(texture1, TexCoord);"
	"\n}";

	unsigned int fragment_program;
	unsigned int vertex_program;
	unsigned int id;
	
	// Fragment shader
	fragment_program = glCreateShader(GL_FRAGMENT_SHADER);
	
	glShaderSource(fragment_program, 1, &fragment_shader_code, NULL);
	glCompileShader(fragment_program);
	
	check_gl_compile_errors(fragment_program, "FRAGMENT");

	
	// Vertex shader
	vertex_program = glCreateShader(GL_VERTEX_SHADER);
	
	glShaderSource(vertex_program, 1, &vertex_shader_code, NULL);
	glCompileShader(vertex_program);
	
	check_gl_compile_errors(vertex_program, "VERTEX");

	// Shader program
	id = glCreateProgram();
	
	glAttachShader(id, fragment_program);
	glAttachShader(id, vertex_program);
	
	glLinkProgram(id);
	
	check_gl_compile_errors(id, "PROGRAM");
	
	// Free shaders
	glDeleteShader(fragment_program);
	glDeleteShader(vertex_program);

	return id;
}
	
void check_gl_compile_errors(unsigned int shader, const std::string &type)
{
	char ilog[1024];
	int success;

	if (type != "PROGRAM") {
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

		if (!success) {
			glGetShaderInfoLog(shader, 1024, NULL, ilog);

			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: "
				<< type << "\n" << ilog
				<< "\n -- -----------------------------"
				"---------------------- -- " << std::endl;
		}
	} else {
		glGetProgramiv(shader, GL_LINK_STATUS, &success);

		if (!success) {
			glGetProgramInfoLog(shader, 1024, NULL, ilog);

			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: "
				<< type << "\n" << ilog
				<< "\n -- ------------------------------"
				"---------------------- -- " << std::endl;
		}
	}
}

}

}
