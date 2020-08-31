#ifndef NODE_H_
#define NODE_H_

// C/C++ headers
#include <vector>

// Engine headers
#include <token.hpp>
#include <label.hpp>
#include <class.hpp>

struct node {
	token *			__tptr;
	lbl			__label;
	cls			__class;
	std::vector <node>	__leaves;
};

#endif