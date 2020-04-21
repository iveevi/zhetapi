#ifndef NODE_H_
#define NODE_H_

/* C++ Standard Libraries */
#include <unordered_map>
#include <functional>
#include <vector>
#include <string>
#include <queue>

/* Engine Headers */
#include "token.h"
#include "operand.h"
#include "operation.h"
#include "variable.h"

template <class T>
class func_stack;

/**
 * @brief The enumeration
 * label is used to label nodes
 * of an expression tree.
 */
enum zlabel {
	l_none,
	l_separable,
	l_multiplied,
	l_divided,
	l_constant,
	l_variable,
	l_polynomial,
	l_exp,
	l_power,
	l_power_uncertain,
	l_exponential,
	l_constant_logarithmic,
	l_operation_constant,
	l_logarithmic,
	l_trigonometric
};

/**
 * @brief String representations
 * of the corresponding labels
 * in the label enumeration.
 */
std::string strlabs[] = {
	"none",
	"separable",
	"multiplied",
	"divided",
	"constant",
	"variable",
	"polynomic",
	"exponent",
	"power",
	"power uncertain",
	"exponential",
	"constant logarithmic",
	"operation constant",
	"logarithmic",
	"trigonometric"
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
	
	using variables = std::unordered_map <std::string,
	      std::vector <node <double> *>>;

	using params = std::vector <variable <double>>;
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
	zlabel type;

	/**
	 * @brief Children of this node
	 * object. There is no fixed
	 * number of children, as the number
	 * of operands of operations varies.
	 */
	std::vector <node *> leaves;

	params pars;
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
	node(std::string, var_stack <T> = var_stack <T> (),
		func_stack <T> = func_stack <T> (),
		params = params(), variables = variables());

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
	node(token *t, std::vector <node *> lv);

	/**
	 * @brief Node constructor, for
	 * all its members.
	 */
	node(token *t, zlabel l, std::vector <node *> lv);

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
	void set(token *, std::vector <node *>);

	/**
	 * @brief Setter method to change
	 * the node's contents without changing
	 * its address. Required by compression
	 * and similar methods in the functor
	 * class.
	 */
	void set(token *, zlabel, std::vector <node *>);

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

	zlabel kind();
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
private:
	/* Helper methods of the
	 * node class, used by
	 * other methods in the
	 * public interface of this
	 * class */

	node *copy() const;
	
	void label_as_operation();

	void compress_as_separable();
	void compress_as_multiplied();
	void compress_as_divided();
	void compress_as_power();

	void differentiate_as_multiplied(const std::string &);
	void differentiate_as_divided(const std::string &);
	void differentiate_as_power(const std::string &);
	void differentiate_as_trigonometric(const std::string &);
	
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

	/* Class aliases, to avoid
	 * potential errors from
	 * using the wrong token type */

	using opn = operation <operand <T>>;
	using opd = operand <T>;

	using var = variable <T>;

	using def = defaults <T>;
	
	/* Augmneted data structures,
	 * used in comperssion methods  */

	struct term_table {
		opd *constant;
		// std::vector
	};
};

#include "../build/parser.tab.c"
#include "../build/lex.yy.c"

/* Constructors */
template <class T>
node <T> ::node(std::string str, var_stack <T> vst,
		func_stack <T> fst, params pars, variables vars)
{
	char *stripped;
	int i;
	
	stripped = new char[str.length() + 1];
	for (i = 0; i < str.length(); i++)
		stripped[i] = str[i];
	stripped[i] = '\n';

	yy_scan_string(stripped);

	node <T> *temp;

	yyparse(temp, pars, vars, vst, fst);

	*this = temp;

	this->pars = pars;
}

template <class T>
node <T> ::node(node *other)
{
	*this = *(other->copy());
	pars = other->pars;
}

template <class T>
node <T> ::node(const node &other)
{
	*this = *(other.copy());
	pars = other->pars;
}

template <class T>
node <T> ::node(token *t, std::vector <node <T> *> lv)
{
	tok = t;
	leaves = lv;
}

template <class T>
node <T> ::node(token *t, zlabel l, std::vector <node <T> *> lv)
	: tok(t), type(l), leaves(lv) {}

/* Setters and Getters */
template <class T>
void node <T> ::set(node *nd)
{
	tok = nd->tok;
	type = nd->type;
	leaves = nd->leaves;
}

template <class T>
void node <T> ::set(token *t, std::vector <node <T> *> lv)
{
	tok = t;
	leaves = lv;
}

template <class T>
void node <T> ::set(token *t, zlabel l, std::vector <node <T> *> lv)
{
	tok = t;
	type = l;
	leaves = lv;
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
zlabel node <T> ::kind()
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
	std::vector <opd> vals;
	
	switch (tok->caller()) {
	case token::OPERAND:
		return (dynamic_cast <opd *> (tok))->get();
	case token::OPERATION:
		for (auto itr : leaves) {
			switch(itr->tok->caller()) {
			case token::OPERAND:
			case token::OPERATION:
				vals.push_back(opd(itr->value()));
				break;
			default:
				throw incomputable_tree();
			}
		}
		
		return (dynamic_cast <opn *> (tok))
			->compute(vals).get();
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
	string sym;

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
	case l_constant:
		tok = new opd(0);
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
		if (tok == &def::add_op)
			return leaves[0]->display() + " + " + leaves[1]->display();

		return leaves[0]->display() + " - " + leaves[1]->display();
	case l_multiplied: 
		return leaves[0]->display() + " * " + leaves[1]->display();
	case l_divided:
		return leaves[0]->display() + " / " + leaves[1]->display();
	case l_constant:
		return tok->str();
	case l_variable:
		return (dynamic_cast <var *> (tok))->symbol();
	// case l_polynomial: Unnecessary label?
	case l_exp:
		return leaves[0]->display() + " ^ " + leaves[1]->display();
	case l_power:
		return leaves[0]->display() + " ^ " + leaves[1]->display();
	// remove operation constant later
	case l_operation_constant:
		return to_string(value());
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
		<< tok->str() << " @ " << this << std::endl;

	counter = 0;
	for (node *itr : leaves) {
		if (itr == nullptr)
			continue;
		itr->print(++counter, lev + 1);
	}
}

/* Helper Methods */
template <class T>
node <T> *node <T> ::copy() const
{
	node *cpy;

	cpy = new node(tok, type, {});
	for (node *nd : leaves)
		cpy->leaves.push_back(nd->copy());
	
	return cpy;
}

template <class T>
void node <T> ::label_as_operation()
{
	size_t i;
	for (i = 0; i < def::m_size; i++) {
		if (def::opers[i].matches((dynamic_cast
				<opn *> (tok))->symbol()))
			break;
	}

	bool constant = true;
	for (node *nd : leaves) {
		if (nd->type != l_constant &&
				type != l_operation_constant) {
			constant = false;
			break;
		}
	}

	if (constant) {
		type = l_operation_constant;
		return;
	}

	switch (i) {
	case def::SINOP:
	case def::COSOP:
	case def::CSCOP:
	case def::SECOP:
	case def::TANOP:
	case def::COTOP:
		type = l_trigonometric;
		break;
	case def::EXPOP:
		if (leaves[0]->type == l_variable) {
			if (leaves[1]->type == l_constant)
				type = l_power;
			else
				// new type
				break;
		} else if (leaves[0]->type == l_constant) {
			if (leaves[1]->type == l_constant)
				type = l_constant;
			else if (leaves[1]->type == l_variable)
				type = l_exponential;
			else
				// something else
				break;
		} else {
			type = l_exp;
		}

		break;
	case def::ADDOP:
	case def::SUBOP:
		type = l_separable;
		break;
	case def::MULTOP:
		type = l_multiplied;
		break;
	case def::DIVOP:
		type = l_divided;
		break;
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
	
	std::vector <opd> vals;

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
				opd(current->value()),
				*(dynamic_cast <opd *> (constant))
			};
			constant = new opd(def::mult_op.compute(vals));
			break;
		case l_constant:
			vals = {
				*(dynamic_cast <opd *> (constant)),
				*(dynamic_cast <opd *> (current->tok))
			};
			constant = new opd(def::mult_op.compute(vals));
			break;
		default:
			misc.push_back(current);
			break;
		}
	}

	/* cout << endl << "CONSTANT:" << endl;
	cout << "\t" << constant->get() << " @ " << constant << endl;

	cout << endl << "VARIABLES:" << endl;
	for (auto itr : chart)
		cout << "\tVAR:\t" << itr.first << ", POW:\t" << itr.second << endl;

	cout << endl << "MISC:" << endl; 
	for (auto nd : misc) {
		cout << std::string(100, '-') << endl;
		compress(nd);
		print(nd, 1, 0);
	}
	
	cout << std::string(100, '-') << endl; */
	
	for (auto nd : misc)
		nd->compress();

	tok = nullptr;
	if (constant->get() == 0) {
		set(new opd(0), {});
		return;
	}

	if (constant->get() != 1)
		set(constant, {});

	for (auto itr : chart) {
		if (tok)
			temp = copy();

		if (itr.second == 1) {
			t = new node(new var(itr.first, true), {});
		} else {
			t = new node(&def::exp_op, {
				new node(new var(itr.first, true), {}),
				new node(new opd(itr.second), {})
			});
		}

		if (tok)
			set(&def::mult_op, {temp, t});
		else
			set(t);
	}

	for (auto itr : misc) {
		if (tok) {
			temp = copy();
			set(&def::mult_op, {temp, itr});
		} else {
			set(itr->tok, itr->leaves);
		}
	}
}

template <class T>
void node <T> ::compress_as_divided()
{

}

template <class T>
void node <T> ::compress_as_power()
{
	T val;

	for (node *nd : leaves)
		nd->compress();

	if (leaves[1]->type == l_constant) {
		val = (dynamic_cast <opd *> (leaves[1]->tok))->get();
		if (val == 1)
			set(leaves[0]);
		else if (val == 0)
			set(new opd(1), {});
	}
}

template <class T>
void node <T> ::differentiate_as_multiplied(const std::string &var)
{
	node *lcpy;
	node *rcpy;

	tok = &def::add_op;

	lcpy = leaves[0]->copy();
	lcpy->differentiate(var);

	rcpy = leaves[1]->copy();
	rcpy->differentiate(var);
	
	leaves = {
		new node(&def::mult_op, {leaves[1], lcpy}),
		new node(&def::mult_op, {leaves[0], rcpy})
	};
}

template <class T>
void node <T> ::differentiate_as_divided(const std::string &var)
{

}

template <class T>
void node <T> ::differentiate_as_power(const std::string &var)
{
	T val;

	val = (dynamic_cast <opd *> (leaves[1]->tok))->get();

	tok = &def::mult_op;

	delete leaves[1];
	leaves[1] = new node(&def::exp_op, {
			new node(leaves[0]->tok, {}),
			new node(new opd(val - 1), {})
	});
	
	leaves[0]->tok = new opd(val);
}

template <class T>
void node <T> ::differentiate_as_trigonometric(const std::string &var)
{
	size_t i;
	for (i = 0; i < def::m_size; i++) {
		if (def::opers[i].matches((dynamic_cast
				<opn *> (tok))->symbol()))
			break;
	}

	switch (i) {
	case def::SINOP:
		tok = &def::cos_op;
		break;
	case def::COSOP:
		set(&def::mult_op, {
			new node(new opd(-1), {}),
			copy()
		});
		break;
	default:
		return;
	}
}

template <class T>
std::string node <T> ::display_as_trigonometric() const
{
	std::string stropn;

	stropn = leaves[0]->display();
	if (tok == &def::sin_op)
		return "sin " + stropn;
	if (tok == &def::cos_op)
		return "cos " + stropn;
	if (tok == &def::tan_op)
		return "tan " + stropn;
	if (tok == &def::csc_op)
		return "csc " + stropn;
	if (tok == &def::sec_op)
		return "sec " + stropn;
	if (tok == &def::cot_op)
		return "cot " + stropn;

	throw node_error("Node labeled as trigonometric, \
		but token is of an undetectable type");
}

#endif
