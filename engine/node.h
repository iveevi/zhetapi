#ifndef NODE_H_
#define NODE_H_

/* C++ Standard Libraries */
#include <vector>
#include <string>

/* Engine Headers */
#include "token.h"

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
	/* Member instances of
	 * the node class */

	token *tok;
	zlabel type;
	std::vector <node *> leaves;
public:
	/* Constructors of the
	 * node class */

	/**
	 * @brief Node constructor, based
	 * on input string. Uses the flex/bison
	 * lexer/parser system to construct tree.
	 */
	node(std::string);

	node(node *);
	node(const node &);

	node(token *t, std::vector <node *> lv);

	/**
	 * @brief Node constructor, for
	 * all its members.
	 */
	node(token *t, zlabel l, std::vector <node *> lv);

	/* Modifiers and getter methods
	 * of the node class */

	/**
	 * @brief Setter method to change
	 * the node's contents without changing
	 * its address. Required by compression
	 * and similar methods in the functor
	 * class.
	 */
	void set(token *, zlabel, std::vector <node *>);

	token *get_token();
	void retokenize(token *);

	node *child_at(size_t);

	zlabel kind();

	/* Functional methods of
	 * the node class */

	bool valid() const;

	const T &value() const;
	
	void label(const std::vector <std::string> &);
	void label_operation();

	void compress();
	void compress_operation();

	void differentiate(const std::string &);
	
	std::string outpt();

	template <class F> 
	void traverse(const F &);

	void traverse(std::function <void (node)> );

	/* Functions mainly for
	 * debugging the node class */

	/**
	 * @brief Node printer. Mainly
	 * used for debugging the
	 * expression tree.
	 */
	void print(int = 1, int = 0);
private:
	/* Helper methods of the
	 * node class, used by
	 * other methods in the
	 * public interface of this
	 * class */

	node *copy();
public:
	/* Exception classes, which
	 * are used in other functions */
	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incomputable_tree {};

	/* Class aliases, to avoid
	 * potential errors from
	 * using the wrong token type */

	using opn = operation <operand <T>>;
	using opd = operand <T>;

	using var = variable <T>;
};

// remove and use
// separate compilation
// instead
#include "parser.tab.c"
#include "lex.yy.c"

/* Constructors */
template <class T>
node <T> ::node(std::string str)
{
	char *stripped;
	int i;
	
	stripped = new char[str.length() + 1];
	for (i = 0; i < str.length(); i++)
		stripped[i] = str[i];
	stripped[i] = '\n';

	yy_scan_string(stripped);

	var_stack <T> vst;
	func_stack <T> fst;

	yyparse(*this, {}, {}, vst, fst);
}

template <class T>
node <T> ::node(token *t, zlabel l, std::vector <node <T> *> lv)
	: tok(t), type(l), leaves(lv) {}

/* Setters and Getters */
template <class T>
void node <T> ::set(token *t, zlabel l, std::vector <node <T> *> lv)
{
	tok = t;
	type = l;
	leaves = lv;
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
				vals.push_back(value(itr));
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
void node <T> ::label(const std::vector <std::string> &vars)
{
	string sym;

	for (auto nd : leaves)
		nd.label();

	switch (tree->tok->caller()) {
	case token::OPERATION:
		label_operation();
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
void node <T> ::label_operation()
{
	size_t i;
	for (i = 0; i < defaults <T> ::m_size; i++) {
		if (defaults <T> ::opers[i].matches
			((dynamic_cast <operation <operand <T>> *>
			(tree->tok))->symbol()))
			break;
	}

	bool constant = true;
	for (node *nd : tree->leaves) {
		if (nd->type != m_constant && nd->type != m_operation_constant) {
			constant = false;
			break;
		}
	}

	if (constant) {
		tree->type = m_operation_constant;
		return;
	}

	if (i >= defaults <T> ::ADDOP && i <= defaults <T> ::MODOP
			&& tree->leaves[0]->type == tree->leaves[1]->type == m_constant) {
		tree->type = m_constant;
		return;
	}

	switch (i) {
	case defaults <T> ::SINOP: case defaults <T> ::COSOP:
	case defaults <T> ::CSCOP: case defaults <T> ::SECOP:
	case defaults <T> ::TANOP: case defaults <T> ::COTOP:
		tree->type = m_trigonometric;
		break;
	case defaults <T> ::EXPOP:
		if (tree->leaves[0]->type == m_variable) {
			if (tree->leaves[1]->type == m_constant)
				tree->type = m_power;
			else
				// new type
				break;
		} else if (tree->leaves[0]->type == m_constant) {
			if (tree->leaves[1]->type == m_constant)
				tree->type = m_constant;
			else if (tree->leaves[1]->type == m_variable)
				tree->type = m_exponential;
			else
				// something else
				break;
		}

		break;
	case defaults <T> ::ADDOP: case defaults <T> ::SUBOP:
		tree->type = m_separable;
		break;
	case defaults <T> ::MULTOP:
		tree->type = m_multiplied;
		break;
	case defaults <T> ::DIVOP:
		tree->type = m_divided;
		break;
	default:
		break;
	}
}

template <class T>
void functor <T> ::compress(node *(&tree))
{
	std::vector <node *> misc;
	std::vector <operand <T>> vals;

	std::string name;
	
	std::queue <node *> que;
	std::unordered_map <std::string, double> chart;
	operand <T> *constant = new operand <T> (1);
	bool var = false;
	node *temp;
	node *t;
	
	/* FIX: treat operation constants seperaely,
	 * outisde of the switch statement; if the
	 * current node is not such (an operation constant),
	 * then compress its children and progress with
	 * checking its specifics
	 * 
	 * REASON: reduce code complexity */
	T val;
	switch (tree->type) {
	case m_operation_constant:
		tree->tok = new operand <T> (value(tree));
		tree->type = functor <T> ::m_constant;
		tree->leaves.clear();
		break;
	case m_power:
		for (node *nd : tree->leaves)
			compress(nd);

		if (tree->leaves[1]->type == m_constant) {
			val = (dynamic_cast <operand <T> *> (tree->leaves[1]->tok))->get();
			if (val == 1) {
				tree->tok = tree->leaves[0]->tok;
				/* if (tree->leaves[1]->type = m_variable) {
					string name = (dynamic_cast <variable <T> *>
							(tree->leaves[1]->tok))->symbol();
					auto itr = m_map[name].begin();
					while (itr != m_map[name].end()) {
						if (*itr == tree->leaves[1]) {
							m_map[name].erase(itr);
							break;
						}
						itr++;
					}
					m_map[name].push_back(tree);
				} */
				
				tree->leaves = tree->leaves[1]->leaves;
				tree->type = tree->leaves[1]->type;
			} else if (val == 0) {
				tree->tok = new operand <double> (1);
				tree->type = m_constant;
				tree->leaves.clear();
			}
		}

		break;
	// add properties of addition/subtraction: x - 0 = x
	// and division: x / 1 = 1
	case m_separable:
		for (node *nd : tree->leaves)
			compress(nd);

		if (tree->leaves[0]->type == m_constant) {
			val = (dynamic_cast <operand <T> *> (tree->leaves[0]->tok))->get();
			if (val == 0) {
				tree->tok = tree->leaves[1]->tok;
				/* if (tree->leaves[1]->type = m_variable) {
					string name = (dynamic_cast <variable <T> *>
							(tree->leaves[1]->tok))->symbol();
					auto itr = m_map[name].begin();
					while (itr != m_map[name].end()) {
						if (*itr == tree->leaves[1]) {
							m_map[name].erase(itr);
							break;
						}
						itr++;
					}
					m_map[name].push_back(tree);
				} */
				
				tree->leaves = tree->leaves[1]->leaves;
				tree->type = tree->leaves[1]->type;
			}
		} else if (tree->leaves[1]->type == m_constant) {
			val = (dynamic_cast <operand <T> *> (tree->leaves[1]->tok))->get();
			if (val == 0) {
				tree->tok = tree->leaves[1]->tok;
				
				/* if (tree->leaves[1]->type = m_variable) {
					string name = (dynamic_cast <variable <T> *> 
							(tree->leaves[1]->tok))->symbol();

					auto itr = m_map[name].begin();
					while (itr != m_map[name].end()) {
						if (*itr == tree->leaves[1]) {
							m_map[name].erase(itr);
							break;
						}
						itr++;
					}
					m_map[name].push_back(tree);
				} */
				
				tree->leaves = tree->leaves[1]->leaves;
				tree->type = tree->leaves[1]->type;
			}
		}

		break;
	case m_multiplied:
		/* FIX: add linear space specific constant, such
		 * as ONE and ZERO
		 *
		 * REASON: allows the client/user to create
		 * different kinds of linear spaces, specifying
		 * what the value of the identity or reflexive
		 * values are */

		/* cout << "ORIGINAL TREE:" << endl;
		cout << std::string(100, '-') << endl;
		print(tree, 1, 0);
		cout << std::string(100, '-') << endl; */

		que.push(tree);

		node *current;
		while (!que.empty()) {
			current = que.front();
			que.pop();

			if (current->type == m_multiplied) {
				que.push(current->leaves[0]);
				que.push(current->leaves[1]);
			} else if (current->type == m_constant) {
				vals.clear();
				vals.push_back(*(dynamic_cast <operand <T> *> (constant)));
				vals.push_back(*(dynamic_cast <operand <T> *> (current->tok)));
				constant = new operand <T> (defaults <T> ::mult_op.compute(vals));
			} else if (current->type == m_operation_constant) {
				vals.clear();
				vals.push_back(operand <T> (value(current)));
				vals.push_back(*(dynamic_cast <operand <T> *> (constant)));
				constant = new operand <T> (defaults <T> ::mult_op.compute(vals));
			} else if (current->type == m_variable) {
				name = (dynamic_cast <variable <T> *> (current->tok))->symbol();
				if (chart.find(name) == chart.end())
					chart[name] = 0;
				chart[name]++;
			} else {
				misc.push_back(current);
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
			compress(nd);

		tree->tok = nullptr;
		if (constant->get() == 0) {
			tree->set(new operand <T> (0), m_constant, {});
			return;
		}

		if (constant->get() != 1)
			tree->set(constant, m_constant, {});

		for (auto itr : chart) {
			if (tree->tok)
				temp = copy(tree);

			if (itr.second == 1) {
				t = new node {new variable <T> {itr.first, true}, m_variable, {}};
			} else {
				t = new node {&defaults <T> ::exp_op, m_power, {
					new node {new variable <T> {itr.first, true}, m_variable, {}},
					new node {new operand <T> {itr.second}, m_constant, {}}
				}};
			}

			if (tree->tok)
				tree->set(&defaults <T> ::mult_op, m_multiplied, {temp, t});
			else
				tree->set(t->tok, t->type, t->leaves);
		}

		for (auto itr : misc) {
			if (tree->tok) {
				temp = copy(tree);
				tree->set(&defaults <T> ::mult_op, m_multiplied, {temp, itr});
			} else {
				tree->set(itr->tok, itr->type, itr->leaves);
			}
		}

		break;
	default:
		for (node *nd : tree->leaves)
			compress(nd);
		break;
	}
}

template <class T>
typename functor <T> ::node *functor <T> ::differentiate
	(node *tree, const std::string &var)
{
	node *left;
	node *right;

	node *lcpy;
	node *rcpy;

	T val;

	switch (tree->type) {
	case m_trigonometric:
		if (tree->tok == &defaults <T> ::sin_op) {
			tree->tok = &defaults <T> ::cos_op;
		} else if (tree->tok == &defaults <T> ::cos_op) {
			tree = new node {&defaults <T> ::mult_op, m_trigonometric, {
				new node {new operand <T> (-1), m_constant, {}},
				tree
			}};
		}

		break;
	case m_power:
		val = (dynamic_cast <operand <T> *> 
			(tree->leaves[1]->tok))->get();

		tree->leaves[1] = new node {&defaults <T> ::opers[defaults <T> ::EXPOP], m_none,
			{new node {tree->leaves[0]->tok, m_variable, {}},
			new node {new operand <T> (val - 1), m_constant, {}}}
		};

		tree->tok = &defaults <T> ::opers[defaults <T> ::MULTOP];
		tree->leaves[0]->tok = new operand <T> (val);
		break;
	case m_separable:
		differentiate(tree->leaves[0], var);
		differentiate(tree->leaves[1], var);
		break;
	case m_multiplied:
		tree->tok = &defaults <T> ::opers[defaults <T> ::ADDOP];

		left = new node {&defaults <T> ::opers[defaults <T> ::MULTOP], m_none, {}};
		right = new node {&defaults <T> ::opers[defaults <T> ::MULTOP], m_none, {}};

		lcpy = copy(tree->leaves[0]);
		rcpy = copy(tree->leaves[1]);

		left->leaves.push_back(differentiate(tree->leaves[0], var));
		left->leaves.push_back(tree->leaves[1]);

		right->leaves.push_back(lcpy);
		right->leaves.push_back(differentiate(rcpy, var));

		tree->leaves[0] = left;
		tree->leaves[1] = right;
		break;
	case m_variable:
		// tree->tok = new operand <T> (1);
		break;
	case m_constant:
		tree->tok = new operand <T> (0);
		break;
	default:
		break;
	}

	return tree;
}

/* Debugging Methods */
template <class T>
void node::print(int num, int lev)
{
	if (nd == nullptr) 
		return;

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
		itr->print(++counter, ++lev);
	}
}

/* Helper Methods */
template <class T>
node <T> *node <T> ::copy()
{
	node *cpy;

	if (tree == nullptr)
		return nullptr;

	cpy = new node {tree->tok, tree->type, {}};
	for (node *nd : tree->leaves)
		cpy->leaves.push_back(copy(nd));
	
	return cpy;
}

#endif
