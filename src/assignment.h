#ifndef ASSIGNMENT_H_
#define ASSIGNMENT_H_

// C++ Standard Libraries
#include <string>

// Custom Built Libraries
#include "var_stack.h"

template <class T>
class assignment {
	std::string m_cached;
public:
	assignment(std::string = "");

	void operator()(var_stack <T> &) const;
};

template <class T>
assignment <T> ::assignment(std::string str) :
	m_cached(str) {}

template <class T>
void assignment <T> ::operator()(var_stack <T> &vst) const
{
	
}

#endif