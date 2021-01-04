#ifndef GLOBAL_H_
#define GLOBAL_H_

// C/C++ headers
#include <iostream>
#include <vector>

// Engine headers
#include <function.hpp>

// Namespaces
using namespace std;

extern size_t line;
extern vector <string> global;
extern zhetapi::Barn <double, int> barn;

extern zhetapi::Operand <bool> *op_true;
extern zhetapi::Operand <bool> *op_false;

int parse();

zhetapi::Token *execute(string);

vector <string> split(string);

zhetapi::Token *print(const std::vector <zhetapi::Token *> &);
zhetapi::Token *println(const std::vector <zhetapi::Token *> &);

#endif
