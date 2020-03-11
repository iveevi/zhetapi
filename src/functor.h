#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C++ Standard Libraries
#include <cstdarg>
#include <string>
#include <vector>
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
class functor {
public:
	/* change naming */
	struct node {
		token *tok;
		std::vector <node *> leaves;
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

	
	/** functor evaluation map : name to nodes*/
	map m_map;
public:
	// if there are no arguments, degrade to
	// lambda (formal token_tree)
	functor(const std::string &);
	functor(const std::string &,
		const std::vector <std::string> &,
		const std::string &);

	const std::string &symbol() const;

	const T &operator()(const std::vector <T> &);

	template <class ... U>
	const T &operator()(U ...);

	void print() const;

	template <class U>
	friend const std::string &output(const functor <U> &);
protected:
	static void print(node *, int, int);

	static node *build(const std::string &, const param_list &, map &);
	static node *build(const std::vector <token *> &, map &);

	static const T &value(const node *);

	template <class ... U>
	static void gather(std::vector <T> &, T, U...);

	static void gather(std::vector <T> &, T);

	enum m_state {
		state_none,
		state_operation,
		state_negative,
		//state_variable,
		//state_operand,
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
	
	m_root = build(expr, m_params, m_map);

	// print();
}

template <class T>
const std::string &functor <T> ::symbol() const
{
	return m_name;
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

		// change size_t to some
		// larger data holder
		if (index == (size_t) -1)
			break;
		
		// dp_var(pr.first->str());

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

// 'Debugging' functors
//
//

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

	std::cout << "#" << num << " - " << nd->tok->str() << " @ " << nd << std::endl;

	counter = 0;
	for (auto it = nd->leaves.begin(); it != nd->leaves.end(); it++, counter++)
		print(*it, counter + 1, lev + 1);
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

// output functions
template <class T>
void process(const typename functor<T>::node *,
	std::string &, size_t);

template <class T>
void process_operation(const typename functor <T> ::node *tree, std::string &str, size_t index)
{
	size_t ind;
	for (size_t i = 0; i < defaults <T> ::m_size; i++) {
		if (defaults <T> ::opers[i].matches((dynamic_cast <operation <operand <T>> *> (tree->tok))->symbol())) {
			ind = i;
			break;
		}
	}

	// dp_var(defaults <T> ::opers[index]);


	// use macros insteead of hardcode and use if ranges
	switch (ind) {
	case 1: case 2: case 3:
	case 4: case 5:
		str.insert(index, " " + defaults <T> ::opers[ind].symbol() + " ");
		process <T> (tree->leaves[0], str, index);
		/*dp_msg("SPLIT--------------------");
		dp_var(tree->tok->str());
		dp_msg("--------------------SPLIT");*/
		process <T> (tree->leaves[1], str, index + 9);
		break;
	case 6: case 7: case 8: case 9:
	case 10: case 11: case 12:
		str.insert(index, " " + defaults <T> ::opers[ind].symbol() + " ");
		process <T> (tree->leaves[0], str, index + 9);
		break;
	}
}

template <class T>
void process(const typename functor <T> ::node *tree, std::string &str, size_t index)
{
	typename functor <T> ::group *g;

	// dp_var(tree->tok->str());
	dp_var(str);

	switch (tree->tok->caller()) {
	case token::OPERATION:
		process_operation <T> (tree, str, index);
		/*str.insert(index, " operation ");
		process <T> (tree->leaves[0], str, index);
		dp_msg("SPLIT--------------------");
		dp_var(tree->tok->str());
		dp_msg("--------------------SPLIT");
		process <T> (tree->leaves[1], str, index + 9);*/
		break;
	case token::FUNCTION:
		// str.insert(index, "operation");
		break;
	case token::OPERAND:
		// str.insert(index, " " + (dynamic_cast <operation <operand <T>> *> (tree->tok))->symbol() + " ");
		break;
	case token::VARIABLE:
		str.insert(index, " variable ");
		break;
	case token::GROUP:
		str.insert(index, " () ");
		g = dynamic_cast <typename functor <T> ::group *> (tree->tok);
		/*if (g != nullptr) {
			dp_msg("is a group");
			dp_var(g->m_root->tok->str());
		} else {
			dp_msg("not a group");
		}*/
		process <T> ((dynamic_cast <typename functor <T> ::group *> (tree->tok))->m_root, str, index + 2);
		break;
	default:
		break;
	}
}

template <class T>
const std::string &output(const functor <T> &func)
{
	std::string out;

	process <T> (const_cast <const typename functor <T> ::node *> (func.m_root), out, 0);

	std::string *nout = new std::string(out);
	return *nout;
}

#endif
