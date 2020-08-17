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
#include "vtable.h"
#include "node.h"

template <class T>
class Vector;

template <class T, class U>
class Function : public token {
public:
	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incompuvtable_tree {};
	
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
	 * Function class */
	Function(const std::string & = "", vtable <T> = vtable <T> ());
	Function(const std::string &, const std::vector <std::string> &,
		const std::string &, vtable <T> = vtable <T> ());

	Function(const std::string &, params, node <T> *);
	Function(const std::string &, stree *, const params &,
			vtable <T> tbl = vtable <T> ());

	Function(const Function &);

	~Function();

	/* Informative functions
	 * and/or methods of the
	 * Function class. */
	size_t ins() const;
	
	const std::string &symbol() const;

	/* Modifiers */
	void rename(const std::string &);
	void rebase(const std::string &, stree *, const params &,
			vtable <T> = vtable <T> ());

	/* Functional methods
	 * of the Function class */
	T compute(const std::vector <T> &);

	const Function &differentiate(const std::string &) const;
	
	const Function &integrate(const std::string &) const;

	std::string display() const;

	nd_class classify();

	/* Virtual functions inehrited
	 * from the token class. */
	std::string str() const override;
	type caller() const override;
	token *copy() const override;

	bool operator==(token *) const override;

	/* Debugging functions */
	void print() const;

	/* Operators for the
	 * Function class */
	const Function &operator=(const Function &);

	const variable <T> &operator[](size_t) const;
	
	T operator()(const Vector <T> &);

	template <class ... U>
	T operator()(U ...);

	template <class A, class B>
	friend const Function <U> &operator+(const Function <U> &,
			const Function <U> &);

	template <class A, class B>
	friend const Function <U> &operator-(const Function <U> &,
			const Function <U> &);

	template <class A, class B>
	friend const Function <U> &operator*(const Function <U> &,
			const Function <U> &);

	template <class A, class B>
	friend const Function <U> &operator/(const Function <U> &,
			const Function <U> &);

	template <class A, class B>
	friend bool operator>(const Function <U> &,
		const Function <U> &);

	template <class A, class B>
	friend bool operator<(const Function <U> &,
		const Function <U> &);

	template <class A, class B>
	friend std::ostream &operator<<(std::ostream &,
		const Function <U> &);
protected:
	template <class ... U>
	static void gather(std::vector <T> &, T, U ...);

	static void gather(std::vector <T> &, T);

	void compress();
	void build();
};

template <class T, class U>
Function <T> ::Function(const std::string &in, vtable <T> tbl)
{
	std::vector <std::string> params;
	std::string name;
	std::string expr;

	// Exit if the function is
	// "supposed" to be empty
	if (in.empty()) {
		m_root = nullptr;
		return;
	}

	// size_t count = 0;
	size_t index;
	for (int i = 0; i < in.length(); i++) {
		if (in[i] == '=') {
			index = i;
			break;
		}
	}

	/* if (count != 1)
		throw invalid_definition(); */

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

template <class T, class U>
Function <T> ::Function(const std::string &str, const std::vector <std::string>
		&names, const std::string &expr, vtable <T> tbl) : m_name(str)
{
	for (auto s : names)
		m_params.push_back(variable <T> (s, true));

	m_root = new node <T> (expr, tbl, m_params);
	compress();
}

template <class T, class U>
Function <T> ::Function(const std::string &str, params pars,
	node <T> *tree)
{
	m_name = str;
	m_params = pars;
	m_root = new node <T> (tree);
	compress();
}

template <class T, class U>
Function <T> ::Function(const std::string &str, stree *raw, const params &pr,
		vtable <T> tbl)
{
	m_name = str;
	m_params = pr;

	m_root = new node <T> (raw, tbl, pr); 
}

template <class T, class U>
Function <T> ::Function(const Function &other) : m_name(other.m_name),
	m_params(other.m_params)
{
	m_root = new node <T> (other.m_root);
	compress();
}

template <class T, class U>
Function <T> ::~Function()
{
	if (m_root)
		delete m_root;
}

template <class T, class U>
size_t Function <T> ::ins() const
{
	return m_params.size();
}

template <class T, class U>
const std::string &Function <T> ::symbol() const
{
	return m_name;
}

template <class T, class U>
void Function <T> ::rename(const std::string &str)
{
	m_name = str;
}

template <class T, class U>
void Function <T> ::rebase(const std::string &str, stree *raw, const params &pr,
		vtable <T> tbl)
{
	m_name = str;
	m_params = pr;

	m_root = new node <T> (raw, tbl, pr); 
}

template <class T, class U>
std::string Function <T> ::display() const
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

template <class T, class U>
nd_class Function <T> ::classify()
{
	m_root->classify();

	return m_root->get_cls();
}

template <class T, class U>
T Function <T> ::compute(const std::vector <T> &vals)
{
	if (vals.size() != m_params.size())
		throw invalid_call();

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_map[m_params[i].symbol()])
			p->retokenize(new operand <T> {vals[i]});
	}

	// Get value, restore tree
	// and return value
	T val = m_root->value();

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_map[m_params[i].symbol()])
			p->retokenize(new variable <T> (m_params[i].symbol(), true));
	}

	return val;
}

template <class T, class U>
const Function <T> &Function <T> ::differentiate
	(const std::string &var) const
{
	node <T> *diffed = new node <T> (m_root);

	diffed->label({var});
	diffed->differentiate(var);
	diffed->reparametrize(m_params);

	Function *out = new Function("d(" + m_name + ")/d(" + var + ")", m_params, diffed);

	delete diffed;

	return *out;
}

template <class T, class U>
const Function <T> &Function <T> ::integrate
	(const std::string &var) const
{
	node <T> *prim = new node <T> (m_root);

	prim->label({var});
	prim->integrate(var);
	prim->reparametrize(m_params);

	Function *out = new Function("d(" + m_name + ")/d(" + var + ")", m_params, prim);

	delete prim;

	return *out;
}

template <class T, class U>
std::string Function <T> ::str() const
{
	return display();
}

template <class T, class U>
token::type Function <T> ::caller() const
{
	return FUNCTOR;
}

template <class T, class U>
token *Function <T> ::copy() const
{
	return new Function(*this); 
}

template <class T, class U>
bool Function <T> ::operator==(token *t) const
{
	if (t->caller() != token::FUNCTOR)
		return false;

	return (m_name == (dynamic_cast <Function *> (t))->m_name)
		&& (m_params == (dynamic_cast <Function *> (t))->m_params);
}

template <class T, class U>
void Function <T> ::print() const
{
	m_root->print();
}

template <class T, class U>
const Function <T> &Function <T> ::operator=(const Function <T> &other)
{
	if (this != &other) {
		m_name = other.m_name;
		m_params = other.m_params;
		m_root = new node <T> (other.m_root);
		compress();
	}

	return *this;
}

template <class T, class U>
const variable <T> &Function <T> ::operator[](size_t i) const
{
	return m_params[i];
}

template <class T, class U>
T Function <T> ::operator()(const Vector <T> &args)
{
	std::vector <T> vals;

	for (size_t i = 0; i < args.size(); i++)
		vals.push_back(args[i]);

	return compute(vals);
}

template <class T, class U>
template <class ... U>
T Function <T> ::operator()(U ... args)
{
	std::vector <T> vals;
	gather(vals, args...);
	return compute(vals);
}

template <class T, class U>
const Function <T> &operator+(const Function <T> &a, const Function <T> &b)
{
	Function <T> *out = new Function <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T> ::params as = a.m_params;
	typename Function <T> ::params bs = b.m_params;

	out->m_params = typename Function <T> ::params(as.size() + bs.size());

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

template <class T, class U>
const Function <T> &operator-(const Function <T> &a, const Function <T> &b)
{
	Function <T> *out = new Function <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T> ::params as = a.m_params;
	typename Function <T> ::params bs = b.m_params;

	out->m_params = typename Function <T> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T> (*a.m_root - *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T, class U>
const Function <T> &operator*(const Function <T> &a, const Function <T> &b)
{
	Function <T> *out = new Function <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T> ::params as = a.m_params;
	typename Function <T> ::params bs = b.m_params;

	out->m_params = typename Function <T> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T> (*a.m_root * *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T, class U>
const Function <T> &operator/(const Function <T> &a, const Function <T> &b)
{
	Function <T> *out = new Function <T> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T> ::params as = a.m_params;
	typename Function <T> ::params bs = b.m_params;

	out->m_params = typename Function <T> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T> (*a.m_root / *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T, class U>
bool operator>(const Function <T> &lhs, const Function <T> &rhs)
{
	return lhs.m_name > rhs.m_name;
}

template <class T, class U>
bool operator<(const Function <T> &lhs, const Function <T> &rhs)
{
	return lhs.m_name < rhs.m_name;
}

template <class T, class U>
std::ostream &operator<<(std::ostream &os, const Function <T> &func)
{
	os << func.display();
	return os;
}

template <class T, class U>
template <class ... U>
void Function <T> ::gather(std::vector <T> &vals,
	T first, U ... args)
{
	vals.push_back(first);
	gather(vals, args...);
}

template <class T, class U>
void Function <T> ::gather(std::vector <T> &vals,
	T first)
{
	vals.push_back(first);
}

template <class T, class U>
void Function <T> ::compress()
{
	m_root->label_all();
	m_root->compress();
	build();
}

template <class T, class U>
void Function <T> ::build()
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

		if (current->get_label() == l_variable) {
			name = (dynamic_cast <variable <T> *> (current->get_token()))->symbol();
			m_map[name].push_back(current);
		}

		for (node <T> *nd : current->get_leaves())
			que.push(nd);
	}
}

#endif
