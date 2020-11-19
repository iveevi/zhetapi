#ifndef PORT_H_
#define PORT_H_

// C/C++ headers
#include <iostream>
#include <vector>

#include <signal.h>
#include <stdlib.h>
#include <string.h>

// Engine headers
#include <function.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <tensor.hpp>

using namespace std;

bool vector_construction_and_memory();
bool matrix_construction_and_memory();
bool tensor_construction_and_memory();

#endif