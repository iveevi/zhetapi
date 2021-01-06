#ifndef GLOBAL_H_
#define GLOBAL_H_

// C/C++ headers
#include <iostream>
#include <vector>

// Engine headers
#include <function.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

extern size_t line;
extern vector <string> global;
extern Barn <double, int> barn;

extern Operand <bool> *op_true;
extern Operand <bool> *op_false;

int parse(char = EOF);
int parse(string);

int compile_library(string);
int assess_library(string);

Token *execute(string);

vector <string> split(string);

Token *print(const vector <Token *> &);
Token *println(const vector <Token *> &);

#endif
