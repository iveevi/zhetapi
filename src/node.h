#ifndef NODE_H
#define NODE_H

// C++ Standard Libraries
#include <vector>

// Custom Built Libraries
#include "list.h"
#include "token.h"
#include "ttwrapper.h"

// remove later (usage of namespaces)
using namespace trees;

// Beginning of node class
template <class T>
struct node {
	ttwrapper <T> *val;
	std::vector <node *> leaves;

	node();
	node(T);

	token *get() const;
	token *operator *() const;
};

template <class T>
node <T> ::node() {}

template <class T>
node <T> ::node(T data) : val(data) {}

template <class T>
token *node <T> ::get() const
{
	return val->get_token();
}

template <class T>
token *node <T> ::operator *() const
{
	return val->get_token();
}

#endif
