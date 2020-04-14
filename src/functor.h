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

// Custom Built Libraries
#include "token.h"
#include "variable.h"
#include "constants.h"
#include "operand.h"
#include "defaults.h"
#include "var_stack.h"

// remove after fixes
#include "debug.h"

/**
 * @brief later add refernces to the vst,
 * and remove copyning refreshing process by
 * changing only token forth and back
 * @tparam T para
 */
template <class T>
class functor : token {
public:
	/* change naming */
	enum m_label {
		m_none,
		m_separable,
		m_multiplied,
		m_divided,
		m_constant,
		m_variable,
		m_polynomial,
		m_power,
		m_exponential,
		m_constant_logarithmic,
		m_operation_constant,
		m_logarithmic,
		m_trigonometric
	};

	static std::string m_label_str[];

	struct node {
		token *tok;
		m_label type;
		std::vector <node *> leaves;

		node(token *t, m_label l, std::vector <node *> lv)
			: tok(t), type(l), leaves(lv) {}
	};

	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incomputable_tree {};
	
	using map = std::unordered_map <std::string, std::vector <node *>>;
	using param_list = std::vector <variable <T>>;
protected:
	/** functor symbol/name */
	std::string m_name;

	/** functors inputs/parameters,
	used really only for checking correctness */
	param_list m_params;

	/** functor tree */
	node *m_root;

	
	/** functor evaluation map : name to nodes */
	map m_map;

	/** functor external evaulation map : for
	 * variables that are not part of m_params */
	map m_emap;
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
	
	void compress();

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
	static void print(node *, int, int);

protected:
	static node *copy(const node *);
	static node *build(const std::string &, const param_list &, map &);
	static node *build(const std::vector <token *> &, map &);

	void label_operation(node *);
	void label(node *, const std::vector <std::string> &);
	void label();

	void compress_operation(node *);
	void compress(node *(&));

	static node *differentiate(node *, const std::string &);

	static bool valid(const node *);
	static const T &value(const node *);

	template <class ... U>
	static void gather(std::vector <T> &, T, U...);

	static void gather(std::vector <T> &, T);

	// for construction from another tree
	void rebuild();
	void rebuild(node *);

	enum m_state {
		state_none,
		state_operation,
		state_negative,
		state_operand,
	};

	static const std::vector <token *> symbols
		(const std::string &, const param_list &,
		map &, var_stack <T> &);

	static const std::pair <token *, size_t>
		&next(const std::string &, const size_t &,
		m_state &, const param_list &, map &,
		var_stack <T> &);

public:
	/* grouping class */
	class group : public token {
	public:
		node *m_root;

		group(node *tree) {
			m_root = tree;
		}

		std::string str() const {
			return "grouper - ()";
		}

		type caller() const {
			return GROUP;
		}
	};
	
	/* replace with vector find
	static size_t get_matching(std::string); */

	friend int yyparse(node *(&), param_list, map &);
};

#include "fparser.h"
#include "lex.yy.c"

// static out of line definitions
template <class T>
std::string functor <T> ::m_label_str[] = {
	"none",
	"separable",
	"multiplied",
	"divided",
	"constant",
	"variable",
	"polynomic",
	"power",
	"exponential",
	"constant logarithmic",
	"operation constant",
	"logarithmic",
	"trigonometric"
};


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
	
	// m_root = build(expr, m_params, m_map);
	char *cpy = new char[expr.length() + 1];
	int i;
	for (i = 0; i < expr.length(); i++)
		cpy[i] = expr[i];
	cpy[i] = '\n';

	yy_scan_string(cpy);
	yyparse(m_root, m_params, m_map);
	compress();

	if (!valid)
		throw invalid_definition();
}

template <class T>
const std::string &functor <T> ::symbol() const
{
	return m_name;
}

template <class T>
typename functor <T> ::node *functor <T> ::copy(const node *tree)
{
	node *cpy;

	if (tree == nullptr)
		return nullptr;

	cpy = new node {tree->tok, tree->type, {}};
	for (node *nd : tree->leaves)
		cpy->leaves.push_back(copy(nd));
	
	return cpy;
}

template <class T>
typename functor <T> ::node *functor <T> ::build
	(const std::string &str, const param_list &m_params, map &m_map)
{
	// dummy
	var_stack <T> vst = var_stack <T> ();
	return build(symbols(str, m_params, m_map, vst), m_map);
}

template <class T>
typename functor <T> ::node *functor <T> ::build
	(const std::vector <token *> &toks, map &m_map)
{
	typedef operation <operand <T>> operation;
	typedef operand <T> operand;

	auto null = defaults <T> ::opers[defaults <T> ::NOPERS];
	auto opers = defaults <T> ::opers;

	node *tree;
	node *temp;

	operation *tptr;
	operation *optr;

	switch (toks.size()) {
	case 0:
		return nullptr;
	case 1:
		tree = new node{toks[0], {}};
		variable <T> *var = dynamic_cast
			<variable <T> *> (toks[0]);

		if (var != nullptr)
			m_map[var->symbol()].push_back(tree);

		return tree;
	}

	auto save = toks.end();
	tptr = &null;
	for (auto it = toks.begin(); it != toks.end(); it++) {
		optr = dynamic_cast <operation *> (*it);
		if (optr != nullptr && optr->get_order()
			<= tptr->get_order()) {
			tptr = optr;
			save = it;
		}
	}

	if (tptr->get_order() == operation::FUNC_LMAX) {
		tree = new node {tptr, {}};

		for (auto it = std::next(save); it != toks.end(); it++) {
			temp = new node {*it, {}};

			variable <T> *var = dynamic_cast
				<variable <T> *> (*it);

			if (var != nullptr)
				m_map[var->symbol()].push_back(temp);

			tree->leaves.push_back(temp);
		}
	} else {
		std::vector <token *> a(toks.begin(), save);
		std::vector <token *> b(std::next(save), toks.end());

		tree = new node{tptr, {}};
		tree->leaves.push_back(build(a, m_map));
		tree->leaves.push_back(build(b, m_map));
	}

	return tree;
}

template <class T>
bool functor <T> ::valid(const node *tree)
{
	typedef operation <operand <T>> operation;
	typedef operand <T> operand;

	std::vector <operand> vals;

	// dp_var(tree->tok->str());
	
	switch (tree->tok->caller()) {
	case token::GROUP:
		return value((dynamic_cast <group *> (tree->tok))->m_root);
	case token::OPERAND:
		return (dynamic_cast <operand *> (tree->tok))->get();
	case token::OPERATION:
		for (auto it = tree->leaves.begin();
			it != tree->leaves.end(); it++) {
			// dp_var((*it)->tok->str());
			switch ((*it)->tok->caller()) {
			case token::OPERAND:
				vals.push_back(*(dynamic_cast <operand *> ((*it)->tok)));
				break;
			case token::OPERATION:
				vals.push_back(operand (value(*it)));
				break;
			case token::GROUP:
				vals.push_back(operand (value((dynamic_cast <group *> ((*it)->tok))->m_root)));
				break;
			default:
				return true;
			}
		}
		
		try {
			(dynamic_cast <operation *> (tree->tok))
				->compute(vals).get();
			return true;
		} catch (...) {
			return false;
		}
	}

	return false;
}

template <class T>
const T &functor <T> ::value(const node *tree)
{
	typedef operation <operand <T>> operation;
	typedef operand <T> operand;

	std::vector <operand> vals;

	// dp_var(tree->tok->str());
	
	switch (tree->tok->caller()) {
	case token::GROUP:
		return value((dynamic_cast <group *> (tree->tok))->m_root);
	case token::OPERAND:
		return (dynamic_cast <operand *> (tree->tok))->get();
	case token::OPERATION:
		for (auto it = tree->leaves.begin();
			it != tree->leaves.end(); it++) {
			// dp_var((*it)->tok->str());
			switch ((*it)->tok->caller()) {
			case token::OPERAND:
				vals.push_back(*(dynamic_cast <operand *> ((*it)->tok)));
				break;
			case token::OPERATION:
				vals.push_back(operand (value(*it)));
				break;
			case token::GROUP:
				vals.push_back(operand (value((dynamic_cast <group *> ((*it)->tok))->m_root)));
				break;
			default:
				throw incomputable_tree();
				break;
			}
		}
		
		return (dynamic_cast <operation *> (tree->tok))
			->compute(vals).get();
	}

	throw incomputable_tree();
}

template <class T>
const std::vector <token *> functor <T> ::symbols(const std::string &str,
	const param_list &params, map &m_map, var_stack <T> &vst)
{
	std::pair <token *, size_t> pr;
	std::vector <token *> toks;
	size_t index = 0;

	// dp_msg("-----------------------------------------------");

	m_state prev;
	
	m_state curr = state_none;
	while (true) {
		prev = curr;

		pr = next(str, index, curr, params, m_map, vst);

		index = pr.second;

		if (pr.first == nullptr)
			break;

		// Assumes that there will be
		// another operand later in the 
		// expression
		if (pr.first != nullptr && pr.first == &defaults <T>
			::opers[defaults <T> ::SUBOP] && prev == state_none) {
			toks.push_back(new operand <T> (-1));
			toks.push_back(&defaults <T>
				::opers[defaults <T> ::MULTOP]);
			continue;
		}
		
		// Assuming operand * operand,
		// where operand is either an
		// actual operand (value), a
		// variable, or a parenthesized
		// sub expression
		if (curr == state_operand) {
			if (prev == curr) {
				toks.push_back(&defaults <T>
					::opers[defaults <T> ::MULTOP]);
			}
		}

		toks.push_back(pr.first);

		// change size_t to some
		// larger data holder
		if (index == (size_t) -1)
			break;
	}

	/* dp_msg("TOKENS");
	for (token *t : toks)
		dp_var(t->str()); */

	return toks;
}

template <class T>
const std::pair <token *, size_t> &functor <T> ::next
	(const std::string &str, const size_t &index,
	m_state &state, const param_list &m_params, map &m_map,
	var_stack <T> &vst)
{
	auto null = defaults <T> ::opers[defaults <T> ::NOPERS];
	auto opers = defaults <T> ::opers;

	size_t m_size = defaults <T> ::m_size;

	std::pair <token *, size_t> *pr = new
		std::pair <token *, size_t> {nullptr, -1};

	std::istringstream ss(str);
	std::string paren;
	std::string accum;
	size_t i;

	T val;

	ss.seekg(index);
	for (i = index; i < str.length(); i++) {
		if (str[i] == '(') {
			for (size_t j = i + 1; j < str.length(); j++) {
				if (str[j] == ')') {
					node *ptree = build(paren, m_params, m_map);

					pr->first = new group(ptree);
					// print(ptree, 0, 0);
					pr->second = j + 1;
					state = state_operand;

					return *pr;
				}

				paren += str[j];
			}
		} else if (str[i] == ')') {
			throw syntax_error();
		}

		if (isdigit(str[i])) {
			ss >> val;

			pr->first = new operand <T> (val);
			pr->second = ss.tellg();
			state = state_operand;

			break;
		}

		if (!isspace(str[i]))
			accum += str[i];

		if (accum.empty())
			continue;

		// Priority with variables over var-stack vars
		auto var = find_if(m_params.begin(), m_params.end(),
			[&](const variable <T> &v) {
				return v.symbol() == accum;
			});
		
		if (var != m_params.end()) {
			pr->first = new variable <T> (var->symbol(), true);
			pr->second = i + 1;
			state = state_operand;

			break;
		}

		auto itr = std::find_if(opers, opers + m_size,
			[&](const operation <operand <T>> &opn) {
				return opn.matches(accum);
			});
		
		if (itr != nullptr && *itr != null) {
			pr->first = itr;
			pr->second = i + 1;
			state = state_operation;

			break;
		}
	}

	return *pr;
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
	label();
	rebuild(m_root);
}

template <class T>
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
}

template <class T>
void functor <T> ::label()
{
	std::vector <std::string> names;
	for (variable <T> v : m_params)
		names.push_back(v.symbol());
	label(m_root, names);
}

template <class T>
void functor <T> ::label_operation(node *tree)
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
void functor <T> ::label(node *tree, const std::vector <std::string> &vars)
{
	string v;
	if (tree == nullptr)
		return;

	for (node *nd : tree->leaves)
		label(nd, vars);

	switch (tree->tok->caller()) {
	case token::OPERATION:
		label_operation(tree);
		break;
	case token::VARIABLE:
		v = (dynamic_cast <variable <T> *>
			(tree->tok))->symbol();
		if (find(vars.begin(), vars.end(), v) != vars.end())
			tree->type = m_variable;
		else
			tree->type = m_constant;
		break;
	case token::OPERAND:
		tree->type = m_constant;
		break;
	}
}

// Beginning of differentiation work

template <class T>
const functor <T> &functor <T> ::differentiate
	(const std::string &var)
{
	functor *out = new functor <T> ();

	node *diffed = copy(m_root);

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

template <class T>
void functor <T> ::compress()
{
	label();
	compress(m_root);
	rebuild();
}

template <class T>
void functor <T> ::compress_operation(node *tree)
{
}

template <class T>
void functor <T> ::compress(node *(&tree))
{
	std::vector <node *> misc;
	std::vector <operand <T>> vals;

	std::string name;
	
	std::queue <node *> que;
	std::unordered_map <std::string, int> chart;
	operand <T> *constant = new operand <T> (1);
	bool var = false;
	node *temp;
	
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
		cout << "ORIGINAL TREE" << endl;
		print(tree, 1, 0);
		for (node *nd : tree->leaves)
			compress(nd);
		cout << "TREE AFTER COMPRESSION OF CHILDREN" << endl;
		print(tree, 1, 0);

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
			} else if (current->type == m_variable) {
				name = (dynamic_cast <variable <T> *> (current->tok))->symbol();
				if (chart.find(name) == chart.end())
					chart[name] = 0;
				chart[name]++;
			} else {
				misc.push_back(tree);
			}
		}

		// cout << "tree @ " << tree << endl;
		// cout << "\tconstant = " << constant->get() << endl;
		for (node *nd : misc) {
			switch (nd->type) {
			case m_variable: case m_power:
			case m_polynomial: 
				var |= true;
				break;
			default:
				break;
			}
		}
		
		tree = new node {new operand <T> (1), m_constant, {}};

		for (auto itr : chart) {
			if (itr.second == 1) {
				temp = new node {new variable <T> {itr.first, true}, m_variable, {}};
			} else {
				temp = new node {&defaults <T> ::exp_op, m_power, {
					new node {new variable <T> {itr.first, true}, m_variable, {}},
					new node {new operand <T> {itr.second}, m_constant, {}}
				}};
			}
			tree = new node {&defaults <T> ::mult_op, m_multiplied, {temp, tree}};
		}

		for (auto nd : misc)
			tree = new node {&defaults <T> ::mult_op, m_multiplied, {nd, tree}};

		if (constant->get() != 1)
			tree = new node {&defaults <T> ::mult_op, m_multiplied, {new node {constant, m_constant, {}}, tree}};
		if (constant->get() == 0)
			tree = new node {new operand <T> (0), m_constant, {}};

		if (tree->type == m_multiplied) {
			if (tree->leaves[0]->type == m_constant) {
				val = (dynamic_cast <operand <T> *> (tree->leaves[0]->tok))->get();
				if (val == 1) {
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
				} else if (val == 0) {
					tree->tok = new operand <double> (0);
					tree->type = m_constant;
					tree->leaves.clear();
				}
			} else if (tree->leaves[1]->type == m_constant) {
				val = (dynamic_cast <operand <T> *> (tree->leaves[1]->tok))->get();
				if (val == 1) {
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
				} else if (val == 0) {
					tree->tok = new operand <double> (0);
					tree->type = m_constant;
					tree->leaves.clear();
				}
			}
		}
		cout << "TREE AFTER COMPRESSION" << endl;
		print(tree, 1, 0);

		break;
	default:
		for (node *nd : tree->leaves)
			compress(nd);
		break;
	}
}

// 'Debugging' functors
//
// Misc functions

template <class T>
void functor <T> ::print() const
{
	print(m_root, 1, 0);
}

template <class T>
void functor <T> ::print(node *nd, int num,
	int lev)
{
	if (nd == nullptr) 
		return;

	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	std::cout << "#" << num << " - [" << m_label_str[nd->type] << "] " << nd->tok->str() << " @ " << nd << std::endl;

	counter = 0;
	for (auto it = nd->leaves.begin(); it != nd->leaves.end(); it++, counter++)
		print(*it, counter + 1, lev + 1);
}

// output functions
template <class T>
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
	case token::GROUP:
		app = "()";
		gr = dynamic_cast <typename functor <T> ::group *> (tree->tok);
		process <T> (gr->m_root, app, 1);
		str.insert(index, app);
		break;
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
}

template <class T>
std::ostream &operator<<(std::ostream &os, const functor <T> &func)
{
	os << output(func);
	return os;
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

#endif
