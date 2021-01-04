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

int parse();

vector <string> split(string);

zhetapi::Token *print(const std::vector <zhetapi::Token *> &);
zhetapi::Token *println(const std::vector <zhetapi::Token *> &);

#endif
