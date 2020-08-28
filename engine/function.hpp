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
#include <node.hpp>

template <class T>
class Vector;

template <class T, class U>
class Function : public token {
public:
	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incompuvtable_tree {};
	
	using Variables = std::unordered_map <std::string,
	      std::vector <node <T, U> *>>;

	using params = std::vector <Variable <T>>;
protected:
	std::string m_name;
	
	params m_params;

	node <T, U> *m_root;
	
	Variables m_map;
public:
	/* Constructors of the
	 * Function class */
	Function(const std::string & = "", vtable <T> = vtable <T> ());
	Function(const std::string &, const std::vector <std::string> &,
		const std::string &, vtable <T> = vtable <T> ());

	Function(const std::string &, params, node <T, U> *);
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

	// return token: provide methods to return coorect (auto) value
	token *compute(const std::vector <node <T, U> *> &);

	const Function &differentiate(size_t) const;
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

	const Variable <T> &operator[](size_t) const;
	
	token *operator()(const Vector <T> &);

	template <class ... V>
	token *operator()(V ...);

	template <class A, class B>
	friend const Function <A, B> &operator+(const Function <A, B> &,
			const Function <A, B> &);

	template <class A, class B>
	friend const Function <A, B> &operator-(const Function <A, B> &,
			const Function <A, B> &);

	template <class A, class B>
	friend const Function <A, B> &operator*(const Function <A, B> &,
			const Function <A, B> &);

	template <class A, class B>
	friend const Function <A, B> &operator/(const Function <A, B> &,
			const Function <A, B> &);

	template <class A, class B>
	friend bool operator>(const Function <A, B> &,
		const Function <A, B> &);

	template <class A, class B>
	friend bool operator<(const Function <A, B> &,
		const Function <A, B> &);

	template <class A, class B>
	friend std::ostream &operator<<(std::ostream &,
		const Function <A, B> &);
protected:
	// change gather
	template <class A, class ... B>
	static void gather(std::vector <node <T, U> *> &, A, B ...);

	template <class A>
	static void gather(std::vector <node <T, U> *> &, A);

	template <class A>
	static node <T, U> *convert(A);

	void compress();
	void build();
};

template <class T, class U>
Function <T, U> ::Function(const std::string &in, vtable <T> tbl)
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
		m_params.push_back(Variable <T> (str, true));

	m_root = new node <T, U> (expr, tbl, m_params);

	compress();
}

template <class T, class U>
Function <T, U> ::Function(const std::string &str, const std::vector <std::string>
		&names, const std::string &expr, vtable <T> tbl) : m_name(str)
{
	for (auto s : names)
		m_params.push_back(Variable <T> (s, true));

	m_root = new node <T, U> (expr, tbl, m_params);
	compress();
}

template <class T, class U>
Function <T, U> ::Function(const std::string &str, params pars,
	node <T, U> *tree)
{
	m_name = str;
	m_params = pars;
	m_root = new node <T, U> (tree);
	compress();
}

template <class T, class U>
Function <T, U> ::Function(const std::string &str, stree *raw, const params &pr,
		vtable <T> tbl)
{
	m_name = str;
	m_params = pr;

	m_root = new node <T, U> (raw, tbl, pr); 
}

template <class T, class U>
Function <T, U> ::Function(const Function &other) : m_name(other.m_name),
	m_params(other.m_params)
{
	m_root = new node <T, U> (other.m_root);
	compress();
}

template <class T, class U>
Function <T, U> ::~Function()
{
	if (m_root)
		delete m_root;
}

template <class T, class U>
size_t Function <T, U> ::ins() const
{
	return m_params.size();
}

template <class T, class U>
const std::string &Function <T, U> ::symbol() const
{
	return m_name;
}

template <class T, class U>
void Function <T, U> ::rename(const std::string &str)
{
	m_name = str;
}

template <class T, class U>
void Function <T, U> ::rebase(const std::string &str, stree *raw, const params &pr,
		vtable <T> tbl)
{
	m_name = str;
	m_params = pr;

	m_root = new node <T, U> (raw, tbl, pr); 
}

template <class T, class U>
std::string Function <T, U> ::display() const
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
nd_class Function <T, U> ::classify()
{
	m_root->classify();

	return m_root->get_cls();
}

template <class T, class U>
token *Function <T, U> ::compute(const std::vector <node <T, U> *> &vals)
{
	if (vals.size() != m_params.size())
		throw invalid_call();

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_map[m_params[i].symbol()])
			p->set(vals[i]);
	}

	// Get value, restore tree
	// and return value
	token *val = m_root->value();

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_map[m_params[i].symbol()])
			p->retokenize(new Variable <T> (m_params[i].symbol(), true));
	}

	return val;
}

template <class T, class U>
const Function <T, U> &Function <T, U> ::differentiate
	(size_t i) const
{
	std::string var = m_params[i].symbol();

	node <T, U> *diffed = new node <T, U> (m_root);

	diffed->label({var});
	diffed->differentiate(var);
	diffed->reparametrize(m_params);

	Function *out = new Function("d(" + m_name + ")/d(" + var + ")", m_params, diffed);

	delete diffed;

	return *out;
}

template <class T, class U>
const Function <T, U> &Function <T, U> ::differentiate
	(const std::string &var) const
{
	node <T, U> *diffed = new node <T, U> (m_root);

	diffed->label({var});
	diffed->differentiate(var);
	diffed->reparametrize(m_params);

	Function *out = new Function("d(" + m_name + ")/d(" + var + ")", m_params, diffed);

	delete diffed;

	return *out;
}

template <class T, class U>
const Function <T, U> &Function <T, U> ::integrate
	(const std::string &var) const
{
	node <T, U> *prim = new node <T, U> (m_root);

	prim->label({var});
	prim->integrate(var);
	prim->reparametrize(m_params);

	Function *out = new Function("d(" + m_name + ")/d(" + var + ")", m_params, prim);

	delete prim;

	return *out;
}

template <class T, class U>
std::string Function <T, U> ::str() const
{
	return display();
}

template <class T, class U>
token::type Function <T, U> ::caller() const
{
	return FUNCTOR;
}

template <class T, class U>
token *Function <T, U> ::copy() const
{
	return new Function(*this); 
}

template <class T, class U>
bool Function <T, U> ::operator==(token *t) const
{
	if (t->caller() != token::FUNCTOR)
		return false;

	return (m_name == (dynamic_cast <Function *> (t))->m_name)
		&& (m_params == (dynamic_cast <Function *> (t))->m_params);
}

template <class T, class U>
void Function <T, U> ::print() const
{
	m_root->print();
}

template <class T, class U>
const Function <T, U> &Function <T, U> ::operator=(const Function <T, U> &other)
{
	if (this != &other) {
		m_name = other.m_name;
		m_params = other.m_params;
		m_root = new node <T, U> (other.m_root);
		compress();
	}

	return *this;
}

template <class T, class U>
const Variable <T> &Function <T, U> ::operator[](size_t i) const
{
	return m_params[i];
}

template <class T, class U>
token *Function <T, U> ::operator()(const Vector <T> &args)
{
	std::vector <T> vals;

	for (size_t i = 0; i < args.size(); i++)
		vals.push_back(args[i]);

	return compute(vals);
}

template <class T, class U>
template <class ... V>
token *Function <T, U> ::operator()(V ... args)
{
	std::vector <node <T, U> *> vals;

	gather(vals, args...);

	return compute(vals);
}

template <class T, class U>
const Function <T, U> &operator+(const Function <T, U> &a, const Function <T, U> &b)
{
	Function <T, U> *out = new Function <T, U> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T, U> ::params as = a.m_params;
	typename Function <T, U> ::params bs = b.m_params;

	out->m_params = typename Function <T, U> ::params(as.size() + bs.size());

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

	out->m_root = new node <T, U> (*a.m_root + *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T, class U>
const Function <T, U> &operator-(const Function <T, U> &a, const Function <T, U> &b)
{
	Function <T, U> *out = new Function <T, U> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T, U> ::params as = a.m_params;
	typename Function <T, U> ::params bs = b.m_params;

	out->m_params = typename Function <T, U> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T, U> (*a.m_root - *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T, class U>
const Function <T, U> &operator*(const Function <T, U> &a, const Function <T, U> &b)
{
	Function <T, U> *out = new Function <T, U> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T, U> ::params as = a.m_params;
	typename Function <T, U> ::params bs = b.m_params;

	out->m_params = typename Function <T, U> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T, U> (*a.m_root * *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T, class U>
const Function <T, U> &operator/(const Function <T, U> &a, const Function <T, U> &b)
{
	Function <T, U> *out = new Function <T, U> ();

	out->m_name = "(" + a.m_name + " + " + b.m_name + ")";

	typename Function <T, U> ::params as = a.m_params;
	typename Function <T, U> ::params bs = b.m_params;

	out->m_params = typename Function <T, U> ::params(as.size() + bs.size());

	std::sort(as.begin(), as.end());
	std::sort(bs.begin(), bs.end());

	auto it = std::set_union(as.begin(), as.end(), bs.begin(), bs.end(),
			out->m_params.begin());

	out->m_params.resize(it - out->m_params.begin());

	out->m_root = new node <T, U> (*a.m_root / *b.m_root);

	out->m_root->set(out->m_params);

	return *out;
}

template <class T, class U>
bool operator>(const Function <T, U> &lhs, const Function <T, U> &rhs)
{
	return lhs.m_name > rhs.m_name;
}

template <class T, class U>
bool operator<(const Function <T, U> &lhs, const Function <T, U> &rhs)
{
	return lhs.m_name < rhs.m_name;
}

template <class T, class U>
std::ostream &operator<<(std::ostream &os, const Function <T, U> &func)
{
	os << func.display();
	return os;
}

template <class T, class U>
template <class A, class ... B>
void Function <T, U> ::gather(std::vector <node <T, U> *> &vals,
	A first, B ... args)
{
	vals.push_back(convert(first));

	gather(vals, args...);
}

template <class T, class U>
template <class A>
void Function <T, U> ::gather(std::vector <node <T, U> *> &vals,
	A first)
{
	vals.push_back(convert(first));
}

template <class T, class U>
template <class A>
node <T, U> *Function <T, U> ::convert(A x)
{
	// Scalars
	if (typeid(A) == typeid(T))
		return new node <T, U> {new operand <T> ((T) x), l_constant_real, {}};
	if (typeid(A) == typeid(U))
		return new node <T, U> {new operand <U> ((U) x), l_constant_integer, {}};
	if (typeid(A) == typeid(Rational <U>))
		return new node <T, U> {new operand <Rational <U>> ((Rational <U>) x), l_constant_rational, {}};
	
	// Complex numbers
	if (typeid(A) == typeid(Complex <T>))
		return new node <T, U> {new operand <Complex <T>> ((T) x), l_constant_complex_real, {}};
	/* if (typeid(A) == typeid(Complex <U>))
		return new {operand <Complex <U>> (x), l_constant_matrix_complex_rational};
	if (typeid(A) == typeid(Complex <Rational <U>>))
		return new operand <Complex <Rational <U>>> (x); */

	// Vectors
	if (typeid(A) == typeid(Vector <T>))
		return new node <T, U> {new operand <Vector <T>> ((Vector <T>) x), l_constant_vector_real, {}};
	if (typeid(A) == typeid(Vector <U>))
		return new node <T, U> {new operand <Vector <Rational <U>>> ((Vector <Rational <U>>) x), l_constant_vector_rational, {}};
	if (typeid(A) == typeid(Vector <Rational <U>>))
		return new node <T, U> {new operand <Vector <Rational <U>>> ((Vector <Rational <U>>) x), l_constant_vector_rational, {}};

	// If no matches, return null
	return nullptr;
}

template <class T, class U>
void Function <T, U> ::compress()
{
	m_root->label_all();
	m_root->compress();
	build();
}

template <class T, class U>
void Function <T, U> ::build()
{
	m_map.clear();
	m_root->label_all();

	std::queue <node <T, U> *> que;
	std::string name;

	node <T, U> *current;

	que.push(m_root);
	while (!que.empty()) {
		current = que.front();
		que.pop();

		if (current->get_label() == l_variable) {
			name = (dynamic_cast <Variable <T> *> (current->get_token()))->symbol();
			m_map[name].push_back(current);
		}

		for (node <T, U> *nd : current->get_leaves())
			que.push(nd);
	}
}

#endif
