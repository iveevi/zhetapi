#ifndef NODE_H
#define NODE_H

// C++ Standard Libraries
#include <vector>

// Custom Built Libraries
#include "list.h"
#include "ttwrapper.h"

// remove later (usage of namespaces)
using namespace trees;

// Beginning of node class
template <typename data_t>
struct node {
	ttwrapper <data_t> *val;
	std::vector <node *> leaves;

	node();
	node(data_t);
};

template <typename data_t>
node <data_t> ::node() {}

template <typename data_t>
node <data_t> ::node(data_t data) : val(data) {}

#endif
