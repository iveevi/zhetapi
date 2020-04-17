#ifndef FUNCTOR_H_
#define FUNCTOR_H_

/* C++ Standard Libraries */
#include <ostream>
#include <cstdarg>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>

/* Engine Headers */
#include "token.h"
#include "variable.h"
#include "constants.h"
#include "operand.h"
#include "defaults.h"
#include "var_stack.h"

#include "node.h"

/**
 * @brief later add refernces to the vst,
 * and remove copyning refreshing process by
 * changing only token forth and back
 * @tparam T para
 */
template <class T>
class functor : token {
public:
	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incomputable_tree {};
	
	using variables = std::unordered_map <std::string,
	      std::vector <node <double> *>>;
	using params = std::vector <variable <double>>;
protected:
	/** functor symbol/name */
	std::string m_name;

	/** functors inputs/parameters,
	used really only for checking correctness */
	params m_params;

	/** functor tree */
	node <T> *m_root;
	
	/** functor evaluation map : name to nodes */
	variables m_map;

	/** functor external evaulation map : for
	 * variables that are not part of m_params *
	map m_emap; */
public:
	// if there are no arguments, degrade to
	// lambda (formal token_tree)
	functor(const std::string & = "");
	functor(const std::string &,
		const std::vector <std::string> &,
		const std::string &);

	const std::string &symbol() const;

	const T &operator()(const std::vector <T> &);

	template <class ... U>
	const T &operator()(U ...);

	const functor &differentiate(const std::string &);

	// The following are helper functions,
	// whether it be output or comparison
	// operators
	
	// void compress();

	void print() const;

	template <class U>
	friend const std::string &output(const functor <U> &);

	/*template <class U>
	size_t process(const typename functor <U> ::node *,
		std::string &, size_t);*/
	
	template <class U>
	friend std::ostream &operator<<(std::ostream &,
		const functor <U> &);

	// Boolean operators
	template <class U>
	friend bool operator>(const functor <U> &,
		const functor <U> &);

	template <class U>
	friend bool operator<(const functor <U> &,
		const functor <U> &);
	
	// restore to private
	// static void print(node *, int, int);
protected:
	/* static node *copy(const node *);
	static node *build(const std::string &, const param_list &, map &);
	static node *build(const std::vector <token *> &, map &); */

	/* void label_operation(node *);
	void label(node *, const std::vector <std::string> &); */
	// void label();

	/* void compress_operation(node *);
	void compress(node *(&)); */

	// static node *differentiate(node *, const std::string &);

	/* static bool valid(const node *);
	static const T &value(const node *); */

	template <class ... U>
	static void gather(std::vector <T> &, T, U...);

	static void gather(std::vector <T> &, T);

	// for construction from another tree
	void rebuild();
	// void rebuild(node *);
/* public:
	friend int yyparse(node *(&), param_list, map &); */
};

//#include "fparser.h"
//#include "lexer.h"
//#include "lex.yy.c"
//#include "function_parser.h"

/* #include "common.h"
#include "function_parser.h"
#include "function_lexer.h" */

/**
 * @brief A constructor which takes in
 * 	the complete definition of the functor
 * 	eg. "f(x) = x^2" [make sure to detect
 * 	variable name collision]
 * 
 * @tparam The base type, with which calculations
 * 	are performed under, eg. int and double
 *
 * @param in The complete definition of the
 *	functor as a string, mentioned @brief
 */
template <class T>
functor <T> ::functor(const std::string &in)
{
	std::vector <std::string> params;
	std::string name;
	std::string expr;

	// Exit if the function is
	// "supposed" to be empty
	if (in.empty())
		return;

	size_t count = 0;
	size_t index;
	for (int i = 0; i < in.length(); i++) {
		if (in[i] == '=') {
			index = i;
			count++;
		}
	}

	if (count != ASSIGNMENT)
		throw invalid_definition();

	name = in.substr(0, index);
	expr = in.substr(index + 1);

	size_t start = -1;
	size_t end;
	for (int i = 0; i < name.length(); i++) {
		if (in[i] == '(' && start == -1)
			start = i;
		if (in[i] == ')')
			end = i;
	}

	std::string vars = name.substr(start + 1, end - start - 1);
	name = name.substr(0, start);

	std::string var;

	for (int i = 0; i < vars.length(); i++) {
		if (vars[i] == ',' && !var.empty()) {
			params.push_back(var);
			var.clear();
		} else if (!isspace(vars[i])) {
			var += vars[i];
		}
	}

	if (!var.empty())
		params.push_back(var);

	m_name = name;

	for (std::string str : params)
		m_params.push_back(variable <T> (str, true));

	// pass necessary info (params, map)
	// once everything else is fixed and
	// dependable
	m_root = new node <T> (expr);
	
	/*
	 * // m_root = build(expr, m_params, m_map);
	char *cpy = new char[expr.length() + 1];
	int i;
	for (i = 0; i < expr.length(); i++)
		cpy[i] = expr[i];
	cpy[i] = '\n';

	f_scan_string(cpy);
	fparse(m_root, m_params, m_map);
	compress(); */

	//if (!valid)
	//	throw invalid_definition();
}

template <class T>
const std::string &functor <T> ::symbol() const
{
	return m_name;
}

template <class T>
const T &functor <T> ::operator()(const std::vector <T> &vals)
{
	if (vals.size() != m_params.size())
		throw invalid_call();

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_map[m_params[i].symbol()])
			p->tok = new operand <T> {vals[i]};
	}

	// Get value, restore tree
	// and return value
	T *val = new T(value(m_root));

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_map[m_params[i].symbol()])
			p->tok = new variable <T> (m_params[i].symbol(), true);
	}

	return *val;
}

template <class T>
template <class ... U>
const T &functor <T> ::operator()(U ... args)
{
	std::vector <T> vals;
	gather(vals, args...);
	return (*this)(vals);
}

template <class T>
template <class ... U>
void functor <T> ::gather(std::vector <T> &vals,
	T first, U ... args)
{
	vals.push_back(first);
	gather(vals, args...);
}

template <class T>
void functor <T> ::gather(std::vector <T> &vals,
	T first)
{
	vals.push_back(first);
}

template <class T>
void functor <T> ::rebuild()
{
	m_map.clear();
	m_root->label();
	m_root->traverse([&](node <T> tree) {
		if (tree.kind() == l_variable) {
			m_map[(dynamic_cast <variable <T> *>
				(tree.get_token()))->symbol()].push_back(tree);
		}
	});
	//label();
	//rebuild(m_root);
}

/* template <class T>
void functor <T> ::rebuild(node *tree)
{
	if (tree == nullptr)
		return;

	if (tree->type == m_variable) {
		m_map[(dynamic_cast <variable <T> *>
			(tree->tok))->symbol()].push_back(tree);
	}

	for (node *nd : tree->leaves)
		rebuild(nd);
} */

/*template <class T>
void functor <T> ::label()
{
	std::vector <std::string> names;
	for (variable <T> v : m_params)
		names.push_back(v.symbol());
	label(m_root, names);
} */

// Beginning of differentiation work

template <class T>
const functor <T> &functor <T> ::differentiate
	(const std::string &var)
{
	functor *out = new functor <T> ();

	node <T> *diffed = new node <T> (m_root);

	label(diffed, {var});

	diffed = differentiate(diffed, var);

	label(diffed, {var});
	compress(diffed);
	std::cout << std::string(50, '-') << std::endl;
	printf("Original:\n");
	print();
	std::cout << std::string(50, '-') << std::endl;

	std::cout << std::string(50, '-') << std::endl;
	printf("Differentiated:\n");
	print(diffed, 1, 0);
	std::cout << std::string(50, '-') << std::endl;

	out->m_root = diffed;
	out->m_name = m_name + "'";
	out->m_params = m_params;

	return *out;
}

/* template <class T>
void functor <T> ::compress()
{
	label();
	compress(m_root);
	rebuild();
}

template <class T>
void functor <T> ::compress_operation(node *tree)
{
} */

// 'Debugging' functors
//
// Misc functions

template <class T>
void functor <T> ::print() const
{
	m_root->print();
}

// output functions
/* template <class T>
size_t process(const typename functor<T>::node *,
	std::string &, size_t);

template <class T>
size_t process_operation(const typename functor <T> ::node *tree, std::string &str, size_t index)
{
	std::string app;

	size_t ind = 0;
	for (size_t i = 0; i < defaults <T> ::m_size; i++) {
		if (defaults <T> ::opers[i].matches((dynamic_cast <operation <operand <T>> *> (tree->tok))->symbol())) {
			ind = i;
			break;
		}
	}

	app = " " + defaults <T> ::opers[ind].symbol() + " ";

	size_t offset = 0;

	// use macros insteead of hardcode and use if ranges
	switch (ind) {
	case 0: case 1: case 2:
	case 3: case 4: case 5:
		str.insert(index, app);
		offset = process <T> (tree->leaves[0], str, index);
		offset += process <T> (tree->leaves[1], str, index + offset + app.length());
		return offset + app.length();
	case 6: case 7: case 8: case 9:
	case 10: case 11: case 12:
		str.insert(index, app);
		return process <T> (tree->leaves[0], str, index + app.length()) + app.length();
	}

	return 0;
}

template <class T>
size_t process(const typename functor <T> ::node *tree, std::string &str, size_t index) // returns offset, length of string appended
{
	typename functor <T> ::group *gr;

	std::string app;
	size_t offset;

	if (tree == nullptr)
		return 0;

	switch (tree->tok->caller()) {
	case token::OPERATION:
		return process_operation <T> (tree, str, index);
	case token::FUNCTION:
		app = "function";
		str.insert(index, app);
		break;
	case token::OPERAND:
		//str.insert(index, " " + std::string((dynamic_cast <operand <T> *> (tree->tok))->symbol()) + " ");
		app = (dynamic_cast <operand <T> *> (tree->tok))->symbol();
		str.insert(index, app);
		break;
	case token::VARIABLE:
		app = (dynamic_cast <variable <T> *> (tree->tok))->symbol();
		str.insert(index, app);
		break;
	* case token::GROUP:
		app = "()";
		gr = dynamic_cast <typename functor <T> ::group *> (tree->tok);
		process <T> (gr->m_root, app, 1);
		str.insert(index, app);
		break; *
	}

	return app.length();
}

template <class T>
const std::string &output(const functor <T> &func)
{
	std::string out;

	process <T> (const_cast <const typename functor <T> ::node *> (func.m_root), out, 0);

	std::string *nout = new std::string();

	*nout += func.m_name + "(";
	for (size_t i = 0; i < func.m_params.size(); i++) {
		*nout += func.m_params[i].symbol();

		if (i < func.m_params.size() - 1)
			*nout += ", ";
	}
	*nout += ") = " + out;

	return *nout;
} */

template <class T>
std::ostream &operator<<(std::ostream &os, const functor <T> &func)
{
	os << output(func);
	return os;
}

// move to table.h
template <class T>
bool operator>(const functor <T> &lhs, const functor <T> &rhs)
{
	return lhs.m_name > rhs.m_name;
}

template <class T>
bool operator<(const functor <T> &lhs, const functor <T> &rhs)
{
	return lhs.m_name < rhs.m_name;
}

#endif
