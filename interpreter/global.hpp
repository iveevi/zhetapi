#ifndef GLOBAL_H_
#define GLOBAL_H_

// Defining the version
#define __zhp_version__ "v1.0.0"

// C/C++ headers
#include <iostream>
#include <vector>
#include <iterator>

// Engine headers
#include "../engine/all/zhplib.hpp"
#include "../engine/core/algorithm.hpp"
#include "../engine/lang/parser.hpp"
#include "../engine/lang/error_handling.hpp"

// Namespaces
using namespace std;
using namespace zhetapi;

extern bool verbose;
extern size_t line;
extern string file;
extern vector <string> global;
extern vector <string> idirs;

extern Engine *engine;

int parse(char = EOF);
int parse(string);

int assess_libraries(Args);
int compile_library(Args, string);

Token *execute(string);

vector <string> split(string);

// Version function
constexpr const char *__get_linted_version__()
{
	return __zhp_version__ " " __TIME__ " " __DATE__;
}

#endif
