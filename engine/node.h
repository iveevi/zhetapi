#ifndef NODE_H_
#define NODE_H_

/* C++ Standard Libraries */
#include <cmath>
#include <cassert>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

/* Engine Headers */
#include "variable.h"
#include "config.h"
#include "stree.h"

template <class T>
class table;

template <class T>
class functor;

/**
 * @brief The enumeration
 * label is used to label nodes
 * of an expression tree.
 *
 * Clarifications:
 *  - l_logarithmic implies a
 *  logarithm of a variable base.
 *  - l_constant_logarithmic implies
 *  a logarithm of a constant base.
 */
enum nd_label {
	l_none,
	l_power,
	l_divided,
	l_variable,
	l_constant,
	l_function,
	l_exp,
	l_polynomial,
	l_separable,
	l_multiplied,
	l_exponential,
	l_logarithmic,
	l_trigonometric,
	l_power_uncertain,
	l_function_constant,
	l_operation_constant,
	l_constant_logarithmic,
	l_summation,
	l_summation_variable,
	l_summation_function
};

/**
 * @brief String representations
 * of the corresponding labels
 * in the label enumeration.
 */
std::string strlabs[] = {
	"none",
	"power",
	"divided",
	"variable",
	"constant",
	"function",
	"exponent",
	"polynomic",
	"separable",
	"multiplied",
	"exponential",
	"logarithmic",
	"trigonometric",
	"power uncertain",
	"function constant",
	"operation constant",
	"constant logarithmic",
	"summation",
	"summation variable",
	"summation function"
};

/**
 * @brief Singular node class,
 * used in expression trees.
 */
template <class T>
class node {
public:
	/* Alias that need to be
	 * defined earlier to be
	 * used later */

	friend class functor <T>;
	
	using params = std::vector <variable <double>>;

	/* Class aliases, to avoid
	 * potential errors from
	 * using the wrong token type */

	using opn = operation <T>;
	using opd = operand <T>;
	using var = variable <T>;
	using ftr = functor <T>;
	using cfg = config <T>;
private:
	/* Member instances of
	 * the node class */

	/**
	 * @brief The main piece of
	 * information being stored in
	 * each node class.
	 */
	token *tok;

	/**
	 * @brief Used to label the node.
	 * This information is used in
	 * more complex functions such
	 * as compression and differentiation
	 */
	nd_label type;

	/**
	 * @brief Children of this node
	 * object. There is no fixed
	 * number of children, as the number
	 * of operands of operations varies.
	 */
	std::vector <node *> leaves;

	/**
	 * @brief List of variable names
	 * that are the names of paramters
	 * (of a function).
	 */
	params pars;

	cfg *cfg_ptr;
public:
	/* Constructors of the
	 * node class */

	/**
	 * @brief Node constructor, based
	 * on input string. Uses the flex/bison
	 * lexer/parser system to construct tree.
	 *
	 * REQUIRES UPDATE ^^
	 */
	node(std::string, table <T> = table <T> (),
			params = params(), cfg * = new cfg());

	/**
	 * @brief Node copy constructor
	 * from a pointer to a node object.
	 */
	node(node *);

	/**
	 * @brief Node copy constructor.
	 */
	node(const node &);

	/**
	 * @brief Node constructor, ignoring
	 * the label of the node. Used in
	 * more complex functions, where
	 * labeling is a post action.
	 */
	node(token *t, std::vector <node *> lv, cfg *);
	
	/**
	 * @brief Node constructor, for
	 * all its members.
	 */
	node(token *t, nd_label l, std::vector <node *> lv, cfg *);

	/* Operators of the node class */
	template <class U>
	friend const node <U> &operator+(const node <U> &, const node <U> &);

	template <class U>
	friend const node <U> &operator-(const node <U> &, const node <U> &);
	
	template <class U>
	friend const node <U> &operator*(const node <U> &, const node <U> &);
	
	template <class U>
	friend const node <U> &operator/(const node <U> &, const node <U> &);

	/* Modifiers and getter methods
	 * of the node class */

	/**
	 * @brief Setter method to transfer
	 * the contents of one node object to
	 * this one. Used to change the representation
	 * of the object without changing its memory
	 * address.
	 */
	void set(node *);

	void set(params);

	/**
	 * @brief Setter method to transfer
	 * content of one node to this node
	 * object, ignoring the label. Used to
	 * simplify node tranfers in more
	 * complex functions where labeling
	 * is pushed to the very end of the process.
	 */
	void set(token *, std::vector <node *>, cfg *);

	/**
	 * @brief Setter method to change
	 * the node's contents without changing
	 * its address. Required by compression
	 * and similar methods in the functor
	 * class.
	 */
	void set(token *, nd_label, std::vector <node *>, cfg *);

	/**
	 * @brief Getter method for
	 * obtaining the token member
	 * of this node object.
	 */
	token *get_token();

	/**
	 * @brief Setter method for
	 * setting or changing the
	 * token member of this
	 * node object.
	 */
	void retokenize(token *);

	void reparametrize(params);

	/**
	 * @brief Getter method to return
	 * the indexed child of this node object.
	 * Used primarily outside this class.
	 */
	node *child_at(size_t);

	std::vector <node *> children();

	nd_label kind();
	// void relabel(label);
	
	/* Functional methods of
	 * the node class */

	bool valid() const;

	const T &value() const;

	void label_all();
	void label(const std::vector <std::string> &);

	void compress();

	void differentiate(const std::string &);

	// template <class F> 
	// void traverse(const F &);

	void traverse(std::function <void (node)> );
	
	std::string display() const;

	/* Functions mainly for
	 * debugging the node class */

	/**
	 * @brief Node printer. Mainly
	 * used for debugging the
	 * expression tree.
	 */
	void print(int = 1, int = 0) const;

	void address_print(int = 1, int = 0) const;
private:
	/* Helper methods of the
	 * node class, used by
	 * other methods in the
	 * public interface of this
	 * class */
	opn *get(opcode) const;
	opn *get(const std::string &) const;

	node *copy() const;
	token *copy(token *) const;

	node *convert(stree *, table <T> = table <T> ()) const;
	node *convert_operation(stree *, table <T> = table <T> ()) const;
	node *convert_summation(stree *, table <T> = table <T> ()) const;
	node *convert_variable_cluster(stree *, table <T> = table <T> ()) const;

	bool special() const;

	void label_as_operation();
	void label_as_special(const std::vector <std::string> &vars);

	void compress_as_separable();
	void compress_as_multiplied();
	void compress_as_divided();
	void compress_as_power();
	void compress_as_exponential();

	void differentiate_as_multiplied(const std::string &);
	void differentiate_as_divided(const std::string &);
	void differentiate_as_power(const std::string &);
	void differentiate_as_exponential(const std::string &);
	void differentiate_as_trigonometric(const std::string &);
	void differentiate_as_logarithmic(const std::string &);
	void differentiate_as_constant_logarithmic(const std::string &);
	void differentiate_as_function(const std::string &);

	std::string display_as_operand(nd_label) const;
	std::string display_as_trigonometric() const;
	std::string display_as_function() const;
public:
	/* Exception classes, which
	 * are used in other functions */

	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incomputable_tree {};
	
	class node_error {
		std::string str;
	public:
		node_error(std::string s)
			: str(s) {}

		const std::string &what() const {
			return str;
		}
	};
	
	class undefined_symbol : public node_error {
	public:
		undefined_symbol(std::string s)
			: node_error(s) {}
	};

	//class undefined_symbol : public node_error {};
	class fatal_error : public node_error {
		fatal_error(std::string s) :
			node_error(s) {}
	};

	class unlabeled_node : public node_error {
	public:
		unlabeled_node(std::string s) :
			node_error(s) {}
	};

	/* Augmneted data structures,
	 * used in comperssion methods  */

	struct term_table {
		opd *constant;
		// std::vector
	};
};

/* Constructors */
template <class T>
node <T> ::node(std::string str, table <T> tbl, params params, cfg *cptr)
	: pars(params), cfg_ptr(cptr)
{
	stree *st = new stree(str);

	if (str.find("sum") != string::npos) {
		cout << endl << "STREE:" << endl;
		st->print();
	}

	node *out = convert(st, tbl);
	if (str.find("sum") != string::npos) {
		cout << endl << "OUT:" << endl;
		out->print();
	}

	*this = out;
}

template <class T>
node <T> ::node(node *other)
{
	*this = *(other->copy());
	pars = other->pars;
	cfg_ptr = other->cfg_ptr;
}

template <class T>
node <T> ::node(const node &other)
{
	*this = *(other.copy());
	pars = other.pars;
	cfg_ptr = other.cfg_ptr;
}

template <class T>
node <T> ::node(token *t, std::vector <node <T> *> lv, cfg *cptr)
{
	tok = t;
	leaves = lv;
	cfg_ptr = cptr;
	type = l_none;
}

template <class T>
node <T> ::node(token *t, nd_label l, std::vector <node <T> *> lv, cfg *cptr)
	: tok(t), type(l), leaves(lv), cfg_ptr(cptr) {}

/* Operators
 *
 * [NOTE]: Assumes
 * both operands have
 * the same cfg_ptr. */
template <class T>
const node <T> &operator+(const node <T> &a, const node <T> &b)
{
	node <T> *out = new node <T> (a.get(op_add), {
			new node <T> (a),
			new node <T> (b)
	}, a.cfg_ptr);

	return *out;
}

template <class T>
const node <T> &operator-(const node <T> &a, const node <T> &b)
{
	node <T> *out = new node <T> (a.get(op_sub), {
			new node <T> (a),
			new node <T> (b)
	}, a.cfg_ptr);

	return *out;
}

template <class T>
const node <T> &operator*(const node <T> &a, const node <T> &b)
{
	node <T> *out = new node <T> (a.get(op_mul), {
			new node <T> (a),
			new node <T> (b)
	}, a.cfg_ptr);

	return *out;
}

template <class T>
const node <T> &operator/(const node <T> &a, const node <T> &b)
{
	node <T> *out = new node <T> (a.get(op_div), {
			new node <T> (a),
			new node <T> (b)
	}, a.cfg_ptr);

	return *out;
}

/* Setters and Getters */
template <class T>
void node <T> ::set(node *nd)
{
	tok = nd->tok;
	type = nd->type;
	leaves = nd->leaves;
	cfg_ptr = nd->cfg_ptr;
}

template <class T>
void node <T> ::set(params p)
{
	pars = p;
}

template <class T>
void node <T> ::set(token *t, std::vector <node <T> *> lv, cfg *cptr)
{
	tok = t;
	leaves = lv;
	cfg_ptr = cptr;
	type = l_none;
}

template <class T>
void node <T> ::set(token *t, nd_label l, std::vector <node <T> *> lv, cfg *cptr)
{
	tok = t;
	type = l;
	leaves = lv;
	cfg_ptr = cptr;
}

template <class T>
token *node <T> ::get_token()
{
	return tok;
}

template <class T>
void node <T> ::retokenize(token *t)
{
	tok = t;
}

template <class T>
void node <T> ::reparametrize(params pr)
{
	pars = pr;
}

template <class T>
node <T> *node <T> ::child_at(size_t i)
{
	return leaves[i];
}

template <class T>
std::vector <node <T> *> node <T> ::children()
{
	return leaves;
}

template <class T>
nd_label node <T> ::kind()
{
	return type;
}

/* Functional Methods */
template <class T>
bool node <T> ::valid() const
{
	size_t operands = 0;
	
	switch (tok->caller()) {
	case token::OPERAND:
		return true;
	case token::OPERATION:
		for (auto itr : leaves) {
			switch(itr->tok->caller()) {
			case token::OPERAND:
				operands++;
				break;
			case token::OPERATION:
				if (!itr->valid())
					return false;
				operands++;
				break;
			default:
				return false;
			}
		}

		if (operands != (dynamic_cast <opn *> (tok))->operands())
			return false;
	}

	return true;
}

template <class T>
const T &node <T> ::value() const
{
	std::vector <token *> vals;
	std::vector <T> hard;

	opn *optr;
	
	switch (tok->caller()) {
	case token::OPERAND:
		return (dynamic_cast <opd *> (tok))->get();
	case token::OPERATION:
		optr = dynamic_cast <opn *> (tok);

		switch (cfg_ptr->code(optr->fmt())) {
		case op_sum:
			for (auto itr : leaves)
				vals.push_back(itr->tok);
			return (new opd((*optr)(vals)))->get();
		}

		for (auto itr : leaves)
			vals.push_back(new opd(itr->value()));
		
		return (new opd((*optr)(vals)))->get();
	case token::VARIABLE:
		if (!(dynamic_cast <var *> (tok))->is_param()) 
			return (dynamic_cast <var *> (tok))->get();
		throw incomputable_tree();
	case token::FUNCTOR:
		for (auto itr : leaves)
			hard.push_back(itr->value());
		
		// change later to allow functions to accept
		// different types of tokens for computation
		return (dynamic_cast <ftr *> (tok))->compute(hard);
	}

	throw incomputable_tree();
}

template <class T>
void node <T> ::label_all()
{
	std::vector <std::string> names;
	
	for (auto var : pars)
		names.push_back(var.symbol());

	label(names);
}

template <class T>
void node <T> ::label(const std::vector <std::string> &vars)
{
	std::string sym;
	
	bool constant = true;

	if (special())
		return label_as_special(vars);

	for (auto nd : leaves)
		nd->label(vars);

	switch (tok->caller()) {
	case token::OPERATION:
		label_as_operation();
		break;
	case token::FUNCTOR:
		type = l_function;
		
		for (node *nd : leaves) {
			if (nd->type != l_constant &&
					nd->type != l_operation_constant) {
				constant = false;
				break;
			}
		}
		
		if (constant)
			type = l_function_constant;

		break;
	case token::VARIABLE:
		sym = (dynamic_cast <var *> (tok))->symbol();

		if (find(vars.begin(), vars.end(), sym) != vars.end())
			type = l_variable;
		else
			type = l_constant;

		break;
	case token::OPERAND:
		type = l_constant;
		break;
	}
}

template <class T>
void node <T> ::compress()
{
	if (type == l_operation_constant
			|| type == l_function_constant) {
		tok = new opd(value());
		type = l_constant;
		leaves.clear();
		return;
	}
	
	switch (type) {
	case l_power:
		compress_as_power();
		break;
	case l_separable:
		compress_as_separable();
		break;
	case l_multiplied:
		compress_as_multiplied();
		break;
	case l_divided:
		compress_as_divided();
		break;
	default:
		for (node *nd : leaves)
			nd->compress();
		break;
	}
}

template <class T>
void node <T> ::differentiate(const std::string &var)
{
	switch (type) {
	case l_constant_logarithmic:
		differentiate_as_constant_logarithmic(var);
		break;
	case l_logarithmic:
		differentiate_as_logarithmic(var);
		break;
	case l_trigonometric:
		differentiate_as_trigonometric(var);
		break;
	case l_power:
		differentiate_as_power(var);
		break;
	case l_exponential:
		differentiate_as_exponential(var);
		break;
	case l_multiplied:
		differentiate_as_multiplied(var);
		break;
	case l_divided:
		differentiate_as_divided(var);
		break;
	case l_function:
		differentiate_as_function(var);
		break;
	case l_separable:
		leaves[0]->differentiate(var);
		leaves[1]->differentiate(var);
		break;
	case l_variable:
		tok = new opd(1);
		break;
	case l_operation_constant:
	case l_constant:
		tok = new opd(0);
		leaves.clear();
		type = l_constant;
		break;
	default:
		break;
	}
}

template <class T>
void node <T> ::traverse(std::function <void (node)> fobj)
{
	fobj(*this);

	for (node *nd : leaves)
		nd->traverse(fobj);
}

template <class T>
std::string node <T> ::display() const
{
	switch (type) {
	case l_separable:
		if (cfg_ptr->code((dynamic_cast <opn *> (tok))->fmt()) == op_add)
			return leaves[0]->display() + " + " + leaves[1]->display();

		return leaves[0]->display() + " - " + leaves[1]->display();
	case l_multiplied: 
		return leaves[0]->display_as_operand(type) + " * " + leaves[1]->display_as_operand(type);
	case l_divided:
		return leaves[0]->display_as_operand(type) + " / " + leaves[1]->display_as_operand(type);
	case l_constant:
		return tok->str();
	case l_variable:
	case l_summation_variable:
		return (dynamic_cast <var *> (tok))->symbol();
	case l_function:
	case l_summation_function:
		return display_as_function();
	// case l_polynomial: Unnecessary label?
	case l_exp:
	case l_power:
	case l_exponential:
		return leaves[0]->display_as_operand(type) + " ^ " + leaves[1]->display_as_operand(type);
	// remove operation constant later
	case l_operation_constant:
		return std::to_string(value());
	case l_constant_logarithmic:
	case l_logarithmic:
		return "log_{" + leaves[0]->display() + "} (" + leaves[1]->display() + ")";
	case l_trigonometric:
		return display_as_trigonometric();
	case l_summation:
		return "sum^{" + leaves[2]->display() + "}_{" + leaves[0]->display() + " = " + leaves[1]->display()
			+ "} " + leaves[3]->display();
	default:
		return "?";
		//throw unlabeled_node("Unlabeled node [" + strlabs[type] + "], could not display it.");
	}
}
/* Debugging Methods */
template <class T>
void node <T> ::print(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	std::cout << "#" << num << " - [" << strlabs[type] << "] "
		<< tok->str() << " @ " << this << endl;

	counter = 0;
	for (node *itr : leaves) {
		if (itr == nullptr)
			continue;
		itr->print(++counter, lev + 1);
	}
}

template <class T>
void node <T> ::address_print(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	std::cout << "#" << num << " - [" << strlabs[type] << "] "
		<< " @ " << this << std::endl;

	counter = 0;
	for (node *itr : leaves) {
		if (itr == nullptr)
			continue;
		itr->address_print(++counter, lev + 1);
	}
}

/* Helper Methods */
template <class T>
typename node <T> ::opn *node <T> ::get(opcode ocode) const
{
	return cfg_ptr->alloc_opn(ocode);
}

template <class T>
typename node <T> ::opn *node <T> ::get(const std::string &str) const
{
	return cfg_ptr->alloc_opn(str);
}

template <class T>
node <T> *node <T> ::copy() const
{
	node *cpy;

	cpy = new node(copy(tok), type, {}, cfg_ptr);
	
	cpy->pars = pars;

	for (node *nd : leaves)
		cpy->leaves.push_back(nd->copy());
	
	return cpy;
}

template <class T>
token *node <T> ::copy(token *t) const
{
	switch (t->caller()) {
	case token::OPERAND:
		return new opd((dynamic_cast <opd *> (t))->get());
	case token::OPERATION:
		return get((dynamic_cast <opn *> (t))->fmt());
	case token::VARIABLE:
		return new var(*(dynamic_cast <var *> (t)));
	case token::FUNCTOR:
		return new ftr(*(dynamic_cast <ftr *> (t)));
	}

	return nullptr;
}

template <class T>
node <T> *node <T> ::convert(stree *st, table <T> tbl) const
{
	node *out;

	T val;

	switch (st->kind()) {
	case l_operation:
		out = convert_operation(st, tbl);
		break;
	case l_variable_cluster:
		out = convert_variable_cluster(st, tbl);
		break;
	case l_number:
		istringstream iss(st->str());

		iss >> val;

		out = new node {new opd(val), l_constant, {}, cfg_ptr};
		break;
	}

	if (out)
		out->pars = pars;

	return out;
}

template <class T>
node <T> *node <T> ::convert_operation(stree *st, table <T> tbl) const
{
	if (st->str() == "sum")
		return convert_summation(st, tbl);

	node *out = new node {get(st->str()), {}, cfg_ptr};
	
	for (stree *s : st->children())
		out->leaves.push_back(convert(s, tbl));

	return out;
}

template <class T>
node <T> *node <T> ::convert_variable_cluster(stree *st, table <T> tbl) const
{
	node *out;

	node *save;
	node *temp;
	node *in;
	
	out = new node {get(op_mul), {
		new node {new opd(1), l_none, {}, cfg_ptr},
		new node {new opd(1), l_none, {}, cfg_ptr}
	}, cfg_ptr};

	temp = out;

	int num = 0;

	std::string str = st->str();
	std::string acc;

	var vr;
	ftr fr;

	for (int i = 0; i < str.length(); i++) {
		acc += str[i];

		auto vitr = find_if(pars.begin(), pars.end(),
			[&](const var &v) {
				return v.symbol() == acc;
			}
		);

		if (vitr != pars.end()) {
			out = new node {get(op_mul), l_none, {
				out,
				new node {new var {vitr->symbol(), true}, {}, cfg_ptr}
			}, cfg_ptr};

			acc.clear();
			num++;
		}

		try {
			vr = tbl.find_var(acc);
			out = new node {get(op_mul), {
				out,
				new node {new var(vr), {}, cfg_ptr}
			}, cfg_ptr};

			acc.clear();
			num++;

			continue;
		} catch(...) {}
		
		try {
			fr = tbl.find_ftr(acc);

			out = new node {get(op_mul), {
				out,
				new node {new ftr(fr), {}, cfg_ptr}
			}, cfg_ptr};

			for (stree *s : st->children())
				out->leaves[1]->leaves.push_back(convert(s, tbl));

			acc.clear();
			num++;

			continue;
		} catch(...) {}
	}

	if (!num) {
		// cout << "Could not disassemble cluster: " << acc << endl;
		throw undefined_symbol(acc);
	}

	return out;
}

template <class T>
node <T> *node <T> ::convert_summation(stree *st, table <T> tbl) const
{
	// Requires 4 operands
	// [remove asserts later]
	assert(st->children().size() == 4);

	string str = st->children()[0]->str();
	string eqn = st->children()[3]->str();

	node *out = new node {get(st->str()), {
		new node {new var {str, T(), true}, {}, cfg_ptr},
		convert(st->children()[1]),
		convert(st->children()[2]),
		new node {new ftr {"f(" + str + ") = " + eqn}, {
			new node {new var {str, T(), true}, {}, cfg_ptr}
		}, cfg_ptr}
	}, cfg_ptr};

	return out;
}

template <class T>
bool node <T> ::special() const
{
	opn *optr;
	switch (tok->caller()) {
	case token::OPERATION:
		optr = dynamic_cast <opn *> (tok);
		switch (cfg_ptr->code(optr->fmt())) {
		case op_sum:
			return true;
		default:
			break;
		}

		break;
	default:
		break;
	}

	return false;
}

template <class T>
void node <T> ::label_as_operation()
{
	bool constant = true;
	for (node *nd : leaves) {
		if (nd->type != l_constant &&
				nd->type != l_operation_constant) {
			constant = false;
			break;
		}
	}
	
	if (constant) {
		type = l_operation_constant;
		return;
	}

	opn *optr = dynamic_cast <opn *> (tok);
	switch (cfg_ptr->code(optr->fmt())) {
	case op_add:
	case op_sub:
		type = l_separable;
		break;
	case op_mul:
		type = l_multiplied;
		break;
	case op_div:
		type = l_divided;
		break;
	case op_exp:
		/* if (leaves[0]->type == l_variable) {
			if (leaves[1]->type == l_constant)
				type = l_power;
			else
				break;
		} else if (leaves[0]->type == l_constant) {
			if (leaves[1]->type == l_constant)
				type = l_constant;
			else if (leaves[1]->type == l_variable)
				type = l_exponential;
			else
				break;
		} else {
			type = l_exp;
		} */

		// If it not an operation constant,
		// it is a power node
		if (leaves[0]->type == l_constant
				|| leaves[0]->type == l_operation_constant)
			type = l_exponential;
		else if (leaves[1]->type == l_constant
				|| leaves[1]->type == l_operation_constant)
			type = l_power;
		break;
	case op_sin:
	case op_cos:
	case op_tan:
	case op_csc:
	case op_sec:
	case op_cot:
		type = l_trigonometric;
		break;
	case op_log:
		if (leaves[0]->type == l_constant)
			type = l_constant_logarithmic;
		else
			type = l_logarithmic;

		break;
	default:
		break;
	}
}

template <class T>
void node <T> ::label_as_special(const std::vector <std::string> &vars)
{
	// Early checking
	opn *optr = dynamic_cast <opn *> (tok);

	if (optr) {
		switch (cfg_ptr->code(optr->fmt())) {
		case op_sum:
			leaves[0]->type = l_summation_variable;
			leaves[1]->label(vars);
			leaves[2]->label(vars);
			leaves[3]->type = l_summation_function;
			leaves[3]->leaves[0]->type = l_summation_variable;
			type = l_summation;
			return;
		default:
			break;
		}
	}
}

template <class T>
void node <T> ::compress_as_separable()
{
	T val;

	for (node *nd : leaves)
		nd->compress();

	if (leaves[0]->type == l_constant) {
		val = (dynamic_cast <opd *> (leaves[0]->tok))->get();
		if (val == 0)
			set(leaves[1]);
	} else if (leaves[1]->type == l_constant) {
		val = (dynamic_cast <opd *> (leaves[1]->tok))->get();
		if (val == 0)
			set(leaves[0]);
	}
}

template <class T>
void node <T> ::compress_as_multiplied()
{
	/* FIX: add linear space specific constant, such
	 * as ONE and ZERO
	 *
	 * REASON: allows the client/user to create
	 * different kinds of linear spaces, specifying
	 * what the value of the identity or reflexive
	 * values are */

	// put the following ds
	// into a single augumented
	// data structure
	std::unordered_map <std::string, T> chart;
	std::vector <node *> misc;
	std::queue <node *> que;
	std::string name;
	
	std::vector <token *> vals;

	opn *optr = get(op_mul);

	opd *constant = new opd(1);
	node *temp, *t;
	node *current;

	que.push(this);

	while (!que.empty()) {
		current = que.front();
		que.pop();

		switch (current->type) {
		case l_multiplied:
			que.push(current->leaves[0]);
			que.push(current->leaves[1]);
			break;
		case l_variable:
			name = (dynamic_cast <var*> (current->tok))->symbol();
			if (chart.find(name) == chart.end())
				chart[name] = 0;
			chart[name]++;
			break;
		case l_operation_constant:
			vals = {
				new opd(current->value()),
				constant
			};
			constant = new opd((*optr)(vals));
			break;
		case l_constant:
			vals = {
				constant,
				current->tok
			};
			constant = new opd((*optr)(vals));
			break;
		default:
			misc.push_back(current);
			break;
		}
	}

	for (auto nd : misc)
		nd->compress();

	tok = nullptr;
	if (constant->get() == 0) {
		set(new opd(0), l_constant, {}, cfg_ptr);
		return;
	}

	if (constant->get() != 1)
		set(constant, l_constant, {}, cfg_ptr);

	for (auto itr : chart) {
		if (tok)
			temp = copy();

		if (itr.second == 1) {
			t = new node(new var(itr.first, true), {}, cfg_ptr);
		} else {
			t = new node(get(op_exp), {
				new node(new var(itr.first, true), {}, cfg_ptr),
				new node(new opd(itr.second), {}, cfg_ptr)
			}, cfg_ptr);
		}

		if (tok) {
			set(cfg_ptr->alloc_opn(op_mul), {temp, t}, cfg_ptr);
		} else {
			set(t);
		}
	}

	for (auto itr : misc) {
		if (tok) {	
			temp = copy();
			set(cfg_ptr->alloc_opn(op_mul), {temp, itr}, cfg_ptr);
		} else {
			set(itr->tok, itr->leaves, cfg_ptr);
		}
	}
}

template <class T>
void node <T> ::compress_as_divided()
{
	T val;
	
	for (node *nd : leaves)
		nd->compress();

	if (leaves[0]->type == l_constant) {
		val = (dynamic_cast <opd *> (leaves[0]->tok))->get();

		if (val == 1) {
			delete tok;
			tok = get(op_exp);

			leaves[0] = leaves[1]->copy();

			delete leaves[1];
			leaves[1] = new node(new opd(-1), {}, cfg_ptr);
		} else if (val == 0) {
			set(new opd(0), {}, cfg_ptr);
		}
	} else if (leaves[1]->type == l_constant) {
		val = (dynamic_cast <opd *> (leaves[1]->tok))->get();

		if (val == cfg_ptr->one) {
			set(leaves[0]);
		}
		/* else if (val == 0) <- throw zero division exception
			set(new opd(1), {}, cfg_ptr); */
	}
}

template <class T>
void node <T> ::compress_as_power()
{
	T val;

	node *lr;
	node *cpy;
	if (leaves[0]->type == l_multiplied) {
		cpy = copy();

		lr = leaves[0]->leaves[1];

		cpy->leaves[0] = lr;
		leaves[0]->leaves[1] = cpy;
		leaves[1] = leaves[0]->leaves[1];
		leaves[0] = leaves[0]->leaves[0];
		tok = get(op_mul);

		leaves[0]->compress();
		leaves[1]->compress();
		
		for (node *nd : leaves)
			nd->compress();
		
		return;
	}

	for (node *nd : leaves)
		nd->compress();
	
	if (leaves[1]->type == l_constant) {
		val = (dynamic_cast <opd *> (leaves[1]->tok))->get();
		if (val == 1)
			set(leaves[0]);
		else if (val == 0)
			set(new opd(1), {}, cfg_ptr);
	}
}

template <class T>
void node <T> ::compress_as_exponential()
{
	T val;

	for (node *nd : leaves)
		nd->compress();

	if (leaves[0]->type == l_constant) {
		val = (dynamic_cast <opd *> (leaves[0]->tok))->get();
		if (val == 1)
			set(new opd(1), {}, cfg_ptr);
		else if (val == 0)
			set(new opd(0), {}, cfg_ptr);
	}
}

template <class T>
void node <T> ::differentiate_as_multiplied(const std::string &var)
{
	node *lcpy;
	node *rcpy;

	tok = get(op_add);

	lcpy = leaves[0]->copy();
	lcpy->differentiate(var);

	rcpy = leaves[1]->copy();
	rcpy->differentiate(var);
	
	leaves = {
		new node(get(op_mul), {lcpy, leaves[1]}, cfg_ptr),
		new node(get(op_mul), {leaves[0], rcpy}, cfg_ptr),
	};
}

template <class T>
void node <T> ::differentiate_as_divided(const std::string &var)
{
	node *denom = leaves[1]->copy();
	node *lcpy = leaves[0]->copy();
	node *rcpy = leaves[1]->copy();

	lcpy->differentiate({var});
	rcpy->differentiate({var});
	
	tok = get(op_div);
	leaves = {
		new node(get(op_sub), {
			new node(get(op_mul), {lcpy, leaves[1]}, cfg_ptr),
			new node(get(op_mul), {leaves[0], rcpy}, cfg_ptr),
		}, cfg_ptr),

		new node(get(op_exp), {
				denom,
				new node(new opd(2), {}, cfg_ptr)
		}, cfg_ptr),
	};
}

template <class T>
void node <T> ::differentiate_as_power(const std::string &var)
{
	T val = (dynamic_cast <opd *> (leaves[1]->tok))->get();

	node *done = leaves[0]->copy();
	done->differentiate({var});
	
	// delete tok;
	tok = get(op_mul);

	// delete leaves[1];
	leaves[1] = new node(get(op_exp), {
			leaves[0],
			new node(new opd(val - 1), {}, cfg_ptr)
	}, cfg_ptr);
	
	// delete leaves[0];
	leaves[0] = new node(get(op_mul), {
			new node(new opd(val), {}, cfg_ptr),
			done
	}, cfg_ptr);
}

template <class T>
void node <T> ::differentiate_as_exponential(const std::string &var)
{
	node *cpy;

	cpy = leaves[1]->copy();
	cpy->differentiate(var);

	T val = (dynamic_cast <opd *> (leaves[0]->tok))->get();

	set(get(op_mul), {
		cpy,
		new node {get(op_mul), {
			new node {new opd(log(val)), {}, cfg_ptr},
			new node {get(op_exp), {
				leaves[0],
				leaves[1]
			}, cfg_ptr}
		}, cfg_ptr}
	}, cfg_ptr);
}

template <class T>
void node <T> ::differentiate_as_trigonometric(const std::string &var)
{
	node *cpy;
	
	cpy = leaves[0]->copy();
	cpy->differentiate(var);

	switch (cfg_ptr->code((dynamic_cast <opn *> (tok))->fmt())) {
	case op_sin:
		set(get(op_mul), {
			cpy,
			new node {get(op_cos), {
				leaves[0]
			}, cfg_ptr},
		}, cfg_ptr);

		break;
	case op_cos:
		set(get(op_mul), {
			new node {new opd(-1), {}, cfg_ptr},
			new node {get(op_mul), {
				cpy,
				new node {get(op_sin), {
					leaves[0]
				}, cfg_ptr},
			}, cfg_ptr}
		}, cfg_ptr);
		break;
	case op_tan:
		set(get(op_mul), {
			cpy,
			new node {get(op_exp), {
				new node {get(op_sec), {
					leaves[0],
				}, cfg_ptr},
				new node {new opd(2), {}, cfg_ptr}
			}, cfg_ptr}
		}, cfg_ptr);
		break;
	case op_csc:
		set(get(op_mul), {
			new node {new opd(-1), {}, cfg_ptr},
			new node {get(op_mul), {
				cpy,
				new node {get(op_mul), {
					new node {get(op_csc), {
						leaves[0]->copy()
					}, cfg_ptr},
					new node {get(op_cot), {
						leaves[0]
					}, cfg_ptr}
				}, cfg_ptr}
			}, cfg_ptr}
		}, cfg_ptr);
		break;
	case op_sec:
		set(get(op_mul), {
			cpy,
			new node {get(op_mul), {
				new node {get(op_sec), {
					leaves[0]->copy()
				}, cfg_ptr},
				new node {get(op_tan), {
					leaves[0]
				}, cfg_ptr}
			}, cfg_ptr}
		}, cfg_ptr);
		break;
	case op_cot:
		set(get(op_mul), {
			new node {new opd(-1), {}, cfg_ptr},
			new node {get(op_mul), {
				cpy,
				new node {get(op_exp), {
					new node {get(op_csc), {
						leaves[0],
					}, cfg_ptr},
					new node {new opd(2), {}, cfg_ptr}
				}, cfg_ptr}
			}, cfg_ptr}
		}, cfg_ptr);
		break;
	default:
		break;
	}
}

template <class T>
void node <T> ::differentiate_as_logarithmic(const std::string &var)
{
}

template <class T>
void node <T> ::differentiate_as_constant_logarithmic(const std::string &var)
{
	T val = log(leaves[0]->value());

	node *cpy = leaves[1]->copy();
	cpy->differentiate(var);

	set(get(op_div), {
		cpy,
		new node {get(op_mul), {
			new node {new opd(val), {}, cfg_ptr},
			leaves[1]
		}, cfg_ptr},
	}, cfg_ptr);
}

template <class T>
void node <T> ::differentiate_as_function(const std::string &var)
{
	// Move using declaration to class
	using ftr = functor <T>;
	
	ftr *ft = dynamic_cast <ftr *> (tok);

	// Remove asserts later
	assert(ft->ins() == leaves.size() && leaves.size() > 0);

	std::vector <node *> terms;

	node *out;
	node *cpy;
	node *lcp;
	
	ftr tmp;

	// First Node
	tmp = ft->differentiate((*ft)[0].symbol());
	
	lcp = copy();
	lcp->retokenize(new ftr(tmp));

	cpy = leaves[0]->copy();

	cpy->differentiate(var);
	// cpy->label_all();

	out = new node {get(op_add), {
		new node {get(op_mul), {
			lcp,
			cpy
		}, cfg_ptr},
		new node {new opd(0), l_none, {}, cfg_ptr}
	}, cfg_ptr};

	// Other parameters
	for (int i = 1; i < ft->ins(); i++) {
		tmp = ft->differentiate((*ft)[i].symbol());

		lcp = copy();
		lcp->retokenize(new ftr(tmp));

		cpy = leaves[i]->copy();

		cpy->differentiate(var);
		// cpy->label_all();

		/* cout << "\t#" << (i + 1) << endl;
		
		cout << "\t\t(a): " << tmp << endl;
		cout << "\t\t(b): " << cpy->display() << endl; */

		out = new node {get(op_add), {
			new node {get(op_mul), {
				lcp,
				cpy
			}, cfg_ptr},
			out
		}, cfg_ptr};
	}

	// cout << endl;

	*this = out->copy();
}

template <class T>
std::string node <T> ::display_as_operand(nd_label required) const
{
	std::string out = display();

	switch (required) {
	case l_trigonometric:
		if (type == l_separable)
			out = "(" + out + ")";
		break;
	case l_exp:
	case l_power:
	case l_exponential:
		if (type != l_constant
			&& type != l_variable)
			out = "(" + out + ")";
		break;
	case l_multiplied:
	case l_divided:
		if (type == l_separable)
			out = "(" + out + ")";
		break;
	default:
		break;
	}

	return out;
}

template <class T>
std::string node <T> ::display_as_trigonometric() const
{
	std::string stropn;

	stropn = leaves[0]->display_as_operand(l_trigonometric);

	switch (cfg_ptr->code((dynamic_cast <opn *> (tok))->fmt())) {
	case op_sin:
		return "sin " + stropn;
	case op_cos:
		return "cos " + stropn;
	case op_tan:
		return "tan " + stropn;
	case op_csc:
		return "csc " + stropn;
	case op_sec:
		return "sec " + stropn;
	case op_cot:
		return "cot " + stropn;
	}

	throw node_error("Node labeled as trigonometric, \
		but token is of an undetectable type");
}

template <class T>
std::string node <T> ::display_as_function() const
{
	std::string out = (dynamic_cast <ftr *> (tok))->symbol();

	out += "(";

	for (size_t i = 0; i < leaves.size(); i++) {
		out += leaves[i]->display();

		if (i < leaves.size() - 1)
			out += ", ";
	}

	out += ")";

	return out;
}

#endif
