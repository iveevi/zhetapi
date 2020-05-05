#ifndef NODE_H_
#define NODE_H_

/* C++ Standard Libraries */
#include <cmath>
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
 */
enum nd_label {
	l_none,
	l_power,
	l_divided,
	l_variable,
	l_constant,
	l_exp,
	l_polynomial,
	l_separable,
	l_multiplied,
	l_exponential,
	l_logarithmic,
	l_trigonometric,
	l_power_uncertain,
	l_operation_constant,
	l_constant_logarithmic
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
	"exponent",
	"polynomic",
	"separable",
	"multiplied",
	"exponential",
	"logarithmic",
	"trigonometric",
	"power uncertain",
	"operation constant",
	"constant logarithmic"
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

	node *convert(stree *, table <T> = table <T> ()) const;
	node *convert_operation(stree *, table <T> = table <T> ()) const;
	node *convert_variable_cluster(stree *, table <T> = table <T> ()) const;

	void label_as_operation();

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

	std::string display_as_operand(nd_label) const;
	std::string display_as_trigonometric() const;
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
	node *out = convert(st, tbl);
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
	pars = other->pars;
	cfg_ptr = other->cfg_ptr;
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
	
	switch (tok->caller()) {
	case token::OPERAND:
		return (dynamic_cast <opd *> (tok))->get();
	case token::VARIABLE:
		if (!(dynamic_cast <var *> (tok))->is_param()) 
			return (dynamic_cast <var *> (tok))->get();
		throw incomputable_tree();
	case token::OPERATION:
		for (auto itr : leaves)
			vals.push_back(new opd(itr->value()));
		
		return (new opd((*(dynamic_cast <opn *> (tok)))(vals)))->get();
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

	for (auto nd : leaves)
		nd->label(vars);

	switch (tok->caller()) {
	case token::OPERATION:
		label_as_operation();
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
	if (type == l_operation_constant) {
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
		return (dynamic_cast <var *> (tok))->symbol();
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

	cpy = new node(tok, type, {}, cfg_ptr);
	for (node *nd : leaves)
		cpy->leaves.push_back(nd->copy());
	
	return cpy;
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
		} catch(...) {}
	}

	if (!num)
		throw undefined_symbol(acc);

	return out;
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
	default:
		break;
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
		new node(get(op_mul), {leaves[1], lcpy}, cfg_ptr),
		new node(get(op_mul), {leaves[0], rcpy}, cfg_ptr),
	};
}

template <class T>
void node <T> ::differentiate_as_divided(const std::string &var)
{

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
	T val = (dynamic_cast <opd *> (leaves[0]->tok))->get();

	val = log(val);

	node *r = leaves[1]->copy();
	
	node *rd = r->copy();

	/* cout << "RIGHT [PRE]" << endl;
	rd->print(); */

	rd->differentiate({var});

	/* cout << "RIGHT [POST]" << endl;
	rd->print(); */

	node *lf = new node(get(op_mul), {
			rd,
			new node(new opd(val), {}, cfg_ptr),
	}, cfg_ptr);

	node *rf = new node(get(op_exp), {
			new node(new opd(exp(1)), {}, cfg_ptr),
			new node(get(op_mul), {
					r,
					new node(new opd(val), {}, cfg_ptr)
			}, cfg_ptr),
	}, cfg_ptr);

	delete tok;
	tok = get(op_mul);

	delete leaves[0];
	leaves[0] = lf;

	delete leaves[1];
	leaves[1] = rf;
}

template <class T>
void node <T> ::differentiate_as_trigonometric(const std::string &var)
{
	switch (cfg_ptr->code((dynamic_cast <opn *> (tok))->fmt())) {
	case op_sin:
		tok = get(op_cos);
		break;
	case op_cos:
		set(get(op_mul), {
			new node(new opd(-1), {}, cfg_ptr),
			copy()
		}, cfg_ptr);
		break;
	default:
		break;
	}
}

template <class T>
std::string node <T> ::display_as_operand(nd_label required) const
{
	std::string out = display();

	switch (required) {
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

	stropn = leaves[0]->display();
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

#endif
