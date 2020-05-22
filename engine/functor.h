#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C++ Standard Libraries
#include <ostream>
#include <cstdarg>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>

// Engine Headers
#include "token.h"
#include "variable.h"
#include "operand.h"
#include "table.h"
#include "node.h"

template <class T>
class functor : public token {
public:
	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incomputable_tree {};
	
	using variables = std::unordered_map <std::string,
	      std::vector <node <T> *>>;

	using params = std::vector <variable <T>>;
protected:
	std::string m_name;
	
	params m_params;

	node <T> *m_root;
	
	variables m_map;
public:
	/* Constructors of the
	 * functor class */
	functor(const std::string & = "", table <T> = table <T> ());
	functor(const std::string &, const std::vector <std::string> &,
		const std::string &);

	functor(const std::string &, params, node <T> *);

	functor(const functor &);

	/* Informative functions
	 * and/or methods of the
	 * functor class. */
	size_t ins() const;

	const std::string &symbol() const;

	/* Functional methods
	 * of the functor class */
	const T &compute(const std::vector <T> &);

	const functor &differentiate(const std::string &) const;

	std::string display() const;

	/* Virtual functions inehrited
	 * from the token class. */
	std::string str() const;
:
	type caller() const;

	/* Debugging functions */
	void print() const;

	/* Operators for the
	 * functor class */
	template <class ... U>
	const T &operator()(U ...);

	template <class U>
	friend const functor <U> &operator+(const functor <U> &,
			const functor <U> &);

	template <class U>
	friend const functor <U> &operator-(const functor <U> &,
			const functor <U> &);

	template <class U>
	friend const functor <U> &operator*(const functor <U> &,
			const functor <U> &);

	template <class U>
	friend const functor <U> &operator/(const functor <U> &,
			const functor <U> &);

	template <class U>
	friend bool operator>(const functor <U> &,
		const functor <U> &);

	template <class U>
	friend bool operator<(const functor <U> &,
		const functor <U> &);

	template <class U>
	friend std::ostream &operator<<(std::ostream &,
		const functor <U> &);
protected:
	template <class ... U>
	static void gather(std::vector <T> &, T, U ...);

	static void gather(std::vector <T> &, T);

	void compress();
	void build();
};

template <class T>
functor <T> ::functor(const std::string &in, table <T> tbl)
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

	if (count != 1)
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
	
	m_root = new node <T> (expr, tbl, m_params);

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
functor <T> ::functor(const functor &other) : m_name(other.m_name),
	m_params(other.m_params), m_root(other.m_root)
{
	compress();
}

template <class T>
size_t functor <T> ::ins() const
{
	return m_params.size();
}

template <class T>
const std::string &functor <T> ::symbol() const
{
	return m_name;
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
const T &functor <T> ::compute(const std::vector <T> &vals)
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
std::string functor <T> ::str() const
{
	return display();
}

template <class T>
token::type functor <T> ::caller() const
{
	return FUNCTOR;
}

template <class T>
void functor <T> ::print() const
{
	m_root->print();
}


template <class T>
template <class ... U>
const T &functor <T> ::operator()(U ... args)
{
	std::vector <T> vals;
	gather(vals, args...);
	return compute(vals);
}

template <class T>
const functor <T> &operator+(const functor <T> &a, const functor <T> &b)
{
	functor <T> *out = new functor <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename functor <T> ::params as = a.m_params;
	typename functor <T> ::params bs = b.m_params;

	out->m_params = typename functor <T> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	/* [FIX]: Add merging the parameters
	 * in the arithmetic operators of the
	 * node class; do not do it manually.
	 *
	 * [REASON]: Easily error prone for
	 * other kinds of operations. */

	out->m_root = new node <T> (*a.m_root + *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T>
const functor <T> &operator-(const functor <T> &a, const functor <T> &b)
{
	functor <T> *out = new functor <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename functor <T> ::params as = a.m_params;
	typename functor <T> ::params bs = b.m_params;

	out->m_params = typename functor <T> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T> (*a.m_root - *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T>
const functor <T> &operator*(const functor <T> &a, const functor <T> &b)
{
	functor <T> *out = new functor <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename functor <T> ::params as = a.m_params;
	typename functor <T> ::params bs = b.m_params;

	out->m_params = typename functor <T> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T> (*a.m_root * *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T>
const functor <T> &operator/(const functor <T> &a, const functor <T> &b)
{
	functor <T> *out = new functor <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename functor <T> ::params as = a.m_params;
	typename functor <T> ::params bs = b.m_params;

	out->m_params = typename functor <T> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T> (*a.m_root / *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

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

template <class T>
std::ostream &operator<<(std::ostream &os, const functor <T> &func)
{
	os << func.display();
	return os;
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
void functor <T> ::compress()
{
	m_root->label_all();
	m_root->compress();
	build();
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

#endif
