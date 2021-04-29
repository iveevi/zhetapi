#ifndef GLOBAL_H_
#define GLOBAL_H_

// C/C++ headers
#include <iostream>
#include <vector>
#include <iterator>

// Engine headers
#include <all/zhplib.hpp>

#include <core/algorithm.hpp>

#include <lang/error_handling.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

extern bool verbose;
extern size_t line;
extern string file;
extern vector <string> global;
extern vector <string> idirs;

extern Engine *engine;

extern Operand <bool> *op_true;
extern Operand <bool> *op_false;

int parse(char = EOF);
int parse(string);

int compile_library(vector <string>, string);

int assess_libraries(vector <string>);

int import_library(string);

Token *execute(string);

vector <string> split(string);

// Include builtin
#include "builtin/basic_io.hpp"

#endif
