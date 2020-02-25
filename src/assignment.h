#ifndef ASSIGNMENT_H_
#define ASSIGNMENT_H_

// C++ Standard Libraries
#include <string>
#include <iostream>

// later replace with formatter
using namespace std;

// Custom Built Libraries
#include "var_stack.h"
#include "expression.h"

// remove after finalizing
#include "debug.h"

template <class T>
class assignment {
	std::string m_cached;
public:
	assignment(const std::string & = "");

	// Take cached string, m_cached
	void operator()(var_stack <T> &) const;

	static void resolve(const std::string &, var_stack <T> &);
};

template <class T>
assignment <T> ::assignment(const std::string &str) :
	m_cached(str) {}

/**
 * @brief parses the input and performs
 * necessary actions
 */
template <class T>
void assignment <T> ::operator()(var_stack <T> &vst) const
{
	// Make sure only one equals
	// sign is present
	size_t count = 0;
	size_t index;
	for (int i = 0; i < m_cached.length(); i++) {
		if (m_cached[i] == '=') {
			index = i;
			count++;
		}
	}

	if (count != 1) {
		cout << "Invalid input" << endl;
		return;
	}

	std::string var = m_cached.substr(0, index);
	std::string expr = m_cached.substr(index + 1);

	dp_var(var);
	dp_var(expr);
}

/**
 * @brief parses the input and performs
 * necessary actions, without caching
 */
template <class T>
void assignment <T> ::resolve(const std::string &str,
	var_stack <T> &vst)
{
	// Make sure only one equals
	// sign is present
	size_t count = 0;
	size_t index;
	for (int i = 0; i < str.length(); i++) {
		if (str[i] == '=') {
			index = i;
			count++;
		}
	}

	if (count != 1) {
		cout << "Invalid input" << endl;
		return;
	}

	std::string var = str.substr(0, index);
	std::string expr = str.substr(index + 1);

	// strip var to the first symbol
	std::string clean;
	for (int i = 0; i < var.length(); i++) {
		if (isspace(var[i]))
			break;
		
		clean += var[i];
	}

	var = clean;

	dp_var(var);
	dp_var(expr);

	T val = expression <T> ::in_place_evaluate(expr);

	cout << "\tsetting [" << var << "] to " << val << endl;

	variable <T> *v;
	
	try {
		v = &vst.get(var);
		v->set(val);
	} catch (typename var_stack <T> ::nfe e) {
		// var isnt in vst
		vst.insert(variable <T> {var, val});
		v = &vst.get(var);
	}
}

#endif