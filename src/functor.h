#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C++ Standard Libraries
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

// remove after fixes
#include "debug.h"

/**
 * @brief later add refernces to the vst
 * @tparam T para
 */
template <class T>
class functor {
public:
	struct node {
		token *tok;
		std::vector <node *> leaves;
	};

	class invalid_definition {};
	class syntax_error {};
	class invalid_call {};
	class incomputable_tree {};
protected:
	/** functor symbol/name */
	std::string m_name;

	/** functors inputs/parameters,
	used really only for checking correctness */
	std::vector <variable <T>> m_params;

	/** functor tree */
	node *m_root;

	
	/** functor evaluation map : name to nodes*/
	std::unordered_map <std::string,
		std::vector <node *>> m_vmap;
public:
	functor(const std::string &);
	functor(const std::string &,
		const std::vector <std::string> &,
		const std::string &);

	const T &operator()(const std::vector <T> &);
	const T &operator()(...);

	void print() const;
protected:
	void print(node *, int, int) const;

	node *copy(const node *) const;
	node *build(const std::string &);
	node *build(const std::vector <token *> &);

	const T &value(const node *) const;

	static const std::vector <token *> symbols
		(const std::string &, const std::vector
		<variable <T>> &);

	static const std::pair <token *, size_t>
		&next(const std::string &, const size_t &,
		const std::vector <variable <T>> &);
	
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
	
	m_root = build(expr);
}

template <class T>
typename functor <T> ::node *functor <T> ::copy
	(const node *tree) const
{
	node *cpy;

	if (tree == nullptr)
		return nullptr;
	
	cpy = new node{tree->tok, {}};
	for (node *nd : tree->leaves)
		cpy->leaves.push_back(copy(nd));

	return cpy;
}

template <class T>
typename functor <T> ::node *functor <T> ::build
	(const std::string &str)
{
	return build(symbols(str, m_params));
}

template <class T>
typename functor <T> ::node *functor <T> ::build
	(const std::vector <token *> &toks)
{
	typedef operation <operand <T>> operation;
	typedef operand <T> operand;

	auto null = defaults <T> ::opers[defaults <T> ::NOPERS];
	auto opers = defaults <T> ::opers;

	node *tree;

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
			m_vmap[var->symbol()].push_back(tree);

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

		for (auto it = std::next(save); it != toks.end(); it++)
			tree->leaves.push_back(new node {*it, {}});
	} else {
		std::vector <token *> a(toks.begin(), save);
		std::vector <token *> b(std::next(save), toks.end());

		tree = new node{tptr, {}};
		tree->leaves.push_back(build(a));
		tree->leaves.push_back(build(b));
	}

	return tree;
}

template <class T>
const T &functor <T> ::value(const node *tree) const
{
	typedef operation <operand <T>> operation;
	typedef operand <T> operand;

	std::vector <operand> vals;
	
	switch (tree->tok->caller()) {
	case token::OPERAND:
		return (dynamic_cast <operand *> (tree->tok))->get();
	case token::OPERATION:
		for (auto it = tree->leaves.begin();
			it != tree->leaves.end(); it++) {
			switch ((*it)->tok->caller()) {
			case token::OPERAND:
				vals.push_back(*(dynamic_cast <operand *> ((*it)->tok)));
				break;
			case token::OPERATION:
				vals.push_back(operand (value(*it)));
			default:
				throw incomputable_tree();
			}

			return (dynamic_cast <operation *> (tree->tok))
				->compute(vals).get();
		}
	}

	throw incomputable_tree();
}

template <class T>
const std::vector <token *> functor <T> ::symbols(const std::string &str,
	const std::vector <variable <T>> &params)
{
	std::pair <token *, size_t> pr;
	std::vector <token *> toks;
	size_t index = 0;
	
	while (true) {
		pr = next(str, index, params);

		index = pr.second;

		if (index == (size_t) -1)
			break;

		toks.push_back(pr.first);
	}

	return toks;
}

template <class T>
const std::pair <token *, size_t> &functor <T> ::next
	(const std::string &str, const size_t &index,
	const std::vector <variable <T>> &params)
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
		/* handle parenthesis later
		if (str[i] == '(') {
			for (size_t j = i + 1; j < str.length(); j++) {
				if (str[j] == ')') {

				}
			}
		} else if (str[i] == ')') {
			throw syntax_error();
		} */

		if (isdigit(str[i])) {
			ss >> val;

			pr->first = new operand <T> (val);
			pr->second = ss.tellg();
			break;
		}

		if (!isspace(str[i]))
			accum += str[i];

		if (accum.empty())
			continue;

		// Priority with variables over var-stack vars
		auto var = find_if(params.begin(), params.end(),
			[&](const variable <T> &v) {
				return v.symbol() == accum;
			});
		
		if (var != params.end()) {
			pr->first = new variable <T> (var->symbol(), true);
			pr->second = i + 1;
			break;
		}

		auto itr = find_if(opers, opers + m_size,
			[&](const operation <operand <T>> &opn) {
				return opn.matches(accum);
			});
		
		if (itr != nullptr && *itr != null) {
			pr->first = itr;
			pr->second = i + 1;
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
	std::cout << "PRINTING TREE" << std::endl;
	print(m_root, 1, 0);
}

template <class T>
void functor <T> ::print(node *nd, int num,
	int lev) const
{
	if (nd == nullptr) 
		return;

	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	std::cout << "#" << num << " - " << nd->tok->str() << std::endl;

	counter = 0;
	for (auto it = nd->leaves.begin(); it != nd->leaves.end(); it++, counter++)
		print(*it, counter + 1, lev + 1);
}

template <class T>
const T &functor <T> ::operator()(const std::vector <T> &vals)
{
	if (vals.size() != m_params.size())
		throw invalid_call();
	
	dp_msg("-------------------------------");
	print(m_root, 0, 0);

	node *cpy = copy(m_root);

	dp_msg("-------------------------------");
	print(cpy, 0, 0);

	for (size_t i = 0; i < m_params.size(); i++) {
		for (auto &p : m_vmap[m_params[i].symbol()]) {
			std::vector <node *> lvs = p->leaves;
			p = new node{new operand <T> {vals[0]}, lvs};
			dp_var(p->tok->get());
		}
	}

	dp_msg("-------------------------------");
	print(m_root, 0, 0);

	// Get value, restore tree
	// and return value
	T *val = new T(value(m_root));
	m_root = cpy;
	return *val;
}

#endif
