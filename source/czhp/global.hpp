#ifndef GLOBAL_H_
#define GLOBAL_H_

// C/C++ headers
#include <iostream>
#include <vector>
#include <iterator>

// Engine headers
#include <function.hpp>

#include <core/algorithm.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

extern size_t line;
extern string file;
extern vector <string> global;
extern vector <string> idirs;
extern Barn <double, int> barn;

extern Operand <bool> *op_true;
extern Operand <bool> *op_false;

int parse(char = EOF);
int parse(string);

int compile_library(vector <string>, string);

int assess_libraries(vector <string>);

int import_library(string);

Token *execute(string);

vector <string> split(string);

Token *print(const vector <Token *> &);
Token *println(const vector <Token *> &);

#endif
