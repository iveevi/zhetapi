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

	functor(const std::string &, params, node <T> *);

	size_t variales() const {
		return m_params.size();
	}

	const std::string &symbol() const;

	const T &operator()(const std::vector <T> &);

	template <class ... U>
	const T &operator()(U ...);

	const functor &differentiate(const std::string &) const;

	std::string display() const;

	// debugging
	void print() const;

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
protected:
	template <class ... U>
	static void gather(std::vector <T> &, T, U...);

	static void gather(std::vector <T> &, T);

	void build();

	void compress();
};

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

	var_stack <T> vst = var_stack <T> ();
	func_stack <T> fst = func_stack <T> ();
	m_root = new node <T> (expr, vst, fst, m_params, m_map);
	
	compress();
}

template <class T>
functor <T> ::functor(const std::string &str, params pars,
	node <T> *tree)
{
	m_name = str;
	m_params = pars;
	m_root = tree;

	compress();
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
			p->retokenize(new operand <T> {vals[i]});
	}

	// Get value, restore tree
	// and return value
	T *val = new T(m_root->value());

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_map[m_params[i].symbol()])
			p->retokenize(new variable <T> (m_params[i].symbol(), true));
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
void functor <T> ::build()
{
	m_map.clear();
	m_root->label_all();

	std::queue <node <T> *> que;
	std::string name;

	node <T> *current;

	que.push(m_root);
	while (!que.empty()) {
		current = que.front();
		que.pop();

		if (current->kind() == l_variable) {
			name = (dynamic_cast <variable <T> *> (current->get_token()))->symbol();
			m_map[name].push_back(current);
		}

		for (node <T> *nd : current->children())
			que.push(nd);
	}
}

template <class T>
void functor <T> ::compress()
{
	m_root->label_all();
	m_root->compress();
	build();
}

// Beginning of differentiation work
template <class T>
const functor <T> &functor <T> ::differentiate
	(const std::string &var) const
{
	node <T> *diffed = new node <T> (m_root);

	diffed->label({var});

	diffed->differentiate(var);
	
	functor *out = new functor("d(" + m_name + ")/d(" + var + ")", m_params, diffed);

	return *out;
}

template <class T>
void functor <T> ::print() const
{
	m_root->print();
}

template <class T>
std::string functor <T> ::display() const
{
	std::string out;

	out += m_name + "("; // if it doesnt have a name, generate a random one (deg_#)
	for (size_t i = 0; i < m_params.size(); i++) {
		out += m_params[i].symbol();

		if (i < m_params.size() - 1)
			out += ", ";
	}

	out += ") = " + m_root->display();
	return out;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const functor <T> &func)
{
	os << func.display();
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
