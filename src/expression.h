#ifndef EXPRESSION_H_
#define EXPRESSION_H_

// C++ Standard Libraries
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <stack>
#include <queue>

// Custom Built Libraries
#include "operation.h"
#include "var_stack.h"
#include "defaults.h"
#include "operand.h"
#include "stack.h"
#include "debug.h"

#include "common.h"
#include "expression_parser.h"
#include "expression_lexer.h"

// Future Note:
// Add parsing method
// for const char *
// parameters (overhead
// reduction from string
// class is possible)

template <class T>
class expression {
	// Future Note:
	// change string operations
	// into const char *
	// operations, also
	// change to a hash table
	std::string m_cached;

	// Future Note:
	// Add later:
	// splay_stack <std::string> m_cached;
public:
	expression(std::string = "");

	// Includes caching, change to references
	const T &evaluate(std::string = "", var_stack <T> = var_stack <T> ());

	// Without caching
	static const T &in_place_evaluate(std::string = "", var_stack <T> = var_stack <T> ());

	// Without caching, and with formatting
	static const T&in_place_evaluate_formatted(const char *, ...);

	// User changeable buffer size
	static int BUF_SIZE;

	class invalid_expr {};
	class syntax_error {};
};

template <class T>
int expression <T> ::BUF_SIZE = 1000;

template <class T>
expression <T> ::expression(std::string str)
	: m_cached(str) {}

template <class T>
const T &expression <T> ::evaluate(std::string str, var_stack <T> vst)
{
	if (str.empty() && m_cached.empty())
		throw invalid_expr();

	if (!str.empty())
		m_cached = str;

	//set_input(str.c_str());
	e_scan_string(str.c_str());
	operand <T> *out = new operand <T> ();
	eparse(out);
	return out->get();
}

template <class T>
const T &expression <T> ::in_place_evaluate(std::string str, var_stack <T> vst)
{
	if (str.empty())
		throw invalid_expr();

	// return value(symbols(str, vst));
	// set_input(str.c_str());
	//cout << "dumping contents of str:" << endl;
	//const char *s = str.c_str();
	//while (s)
	//	cout << *s << " @ " << (int)*s++ << endl;
	char *cpy = new char[str.length() + 1];
	int i;
	for (i = 0; i < str.length(); i++)
		cpy[i] = str[i];
	cpy[i] = '\n';

	/* s = cpy;
	while (s)
		cout << *s << " @ " << (int)*s++ << endl; */
	e_scan_string(cpy);
	operand <T> *out = new operand <T> ();
	
	eparse(out, vst);
	return out->get();
}

template <class T>
const T &expression <T> ::in_place_evaluate_formatted(const char *format, ...)
{
	va_list arg;
	int done;

	char buf[BUF_SIZE];
	
	va_start(arg, format);
	done = sprintf(buf, format, arg);
	va_end(arg);

	if (done < 0)
		throw invalid_expr();

	// Future Note:
	// remove overhead
	// with string class
	
	return in_place_evaluate(std::string(buf));
}

/* template <class T>
const T &expression <T> ::value(const std::string &input)
{
	enum states {};

	states state;

	using operation = operation <operand <T>>;
	using operand = operand <T>;

	std::stack <operation *> operation_stack;
	std::stack <operand *> postfix_stack;
	std::queue <token *> postfix_queue;

	operation *dopn;
	operand *dopd;

	std::istringstream ss(input);

	int size = input.size();
	int index = 0;

	T val;

	token *tptr;

	while (index < size) {
		ss.seekg(index);
		while (index < size) {
			if (isdigit(input[index])) {
				ss >> val;

				tptr = new operand <T> (val);
				break;
			}
		}
	}
}

template <class T>
const T &expression <T> ::value(const std::vector <token *> &toks)
{
	* std::stack <operation <operand <T>> *> operator_stack;
	std::stack <operand <T> *> postfix_stack;
	std::queue <token *> postfix_queue;

	token *t;
	stl_reveal(t, toks, [](token *tptr) {
			return (tptr) ? tptr->str() : "nullptr";
	});

	using namespace std;

	for (auto it = toks.begin(); it != toks.end(); it++) {
		cout << "Token: " << (*it)->str() << endl;
		if (dynamic_cast <operand <T> *> (*it)) {
			cout << "\tBranch A (Operand)" << endl;
			postfix_queue.push(*it);
		} else if (dynamic_cast <operation <operand <T>> *> (*it)) {
			if (dynamic_cast <operation <operand <T>> *> (*it)->get_order() == operation <operand <T>> ::FUNC_LMAX) {
				cout << "\tBranch B1 (Operation - Functions)" << endl;

				decltype(it) cit = std::next(it);
				for (int i = 0; i < dynamic_cast <operation <operand <T>> *> (*it)->get_opers(); i++, cit++) {
					if (cit == toks.end())
						throw invalid_expr();
					postfix_queue.push(*cit);
				}

				postfix_queue.push(*it);

				it = cit;
			} else {
				cout << "\tBranch B2 (Operation - Binary Operators)" << endl;
				while (!operator_stack.empty() && operator_stack.top()->get_order() >= dynamic_cast <operation <operand <T>> *> (*it)->get_order()) {
					cout << "\t\tPopping " << operator_stack.top()->str() << endl;
					postfix_queue.push(operator_stack.top());
					operator_stack.pop();
				}

				operator_stack.push(dynamic_cast <operation <operand <T>> *> (*it));
			}
		} else if (dynamic_cast <lparen *> (*it)) {
			cout << "\tBranch C (Left Parenthesis)" << endl;
			operator_stack.push(dynamic_cast <operation <operand <T>> *> (*it));
		} else if (dynamic_cast <rparen *> (*it)) {
			cout << "\tBranch D (Right Parenthesis)" << endl;
			while (!dynamic_cast <lparen *> (operator_stack.top())) {
				postfix_queue.push(operator_stack.top());
				operator_stack.pop();
			}
			
			operator_stack.pop();
		} else {
			cout << "\tBranch E (Exception)" << endl;
			throw invalid_expr();
		}
	}

	std::queue <token *> copy = postfix_queue;
	while (!copy.empty()) {
		if (copy.front())
			std::cout << copy.front()->str() << std::endl;
		else
			std::cout << "nullptr" << std::endl;
		copy.pop();
	}

	std::vector <operand <T>> vals;

	token *tptr, *temp;
	while (!postfix_queue.empty()) {
		tptr = postfix_queue.front();
		postfix_queue.pop();
		
		if (dynamic_cast <operand <T> *> (tptr)) {
			postfix_stack.push(dynamic_cast <operand <T> *> (tptr));
		} else if (dynamic_cast <operation <operand <T>> *> (tptr)) {
			for (int i = 0; i < dynamic_cast <operation <operand <T>> *> (tptr)->get_opers(); i++) {
				if (postfix_queue.empty())
					throw invalid_expr();

				temp = postfix_queue.front();
				postfix_queue.pop();

				if (!dynamic_cast <operand <T> *> (temp))
					throw invalid_expr();

				vals.push_back(*(dynamic_cast <operand <T> *> (tptr)));
			}

			postfix_stack.push(new operand <T> (dynamic_cast <operation <operand <T>> *> (tptr)->compute(vals)));
		} else {
			throw invalid_expr();
		}
	}

	const int VALUES = 1;
	if (postfix_stack.size() != VALUES)
		throw invalid_expr();
	return postfix_stack.top()->get(); *
}

* template <class T>
const std::vector <token *> &expression <T> ::symbols(const std::string &str,
	var_stack <T> &vst)
{
	std::vector <token *> *toks = new std::vector <token *> ();

	std::pair <token *, size_t> pr;
	size_t index = 0;

	// dp_msg("-----------------------------------------------");

	m_state prev;
	
	m_state curr = state_none;
	while (true) {
		prev = curr;

		pr = next(str, index, curr, vst);

		index = pr.second;

		if (pr.first == nullptr)
			break;

		// Assumes that there will be
		// another operand later in the 
		// expression
		if (pr.first != nullptr && pr.first == &defaults <T>
			::opers[defaults <T> ::SUBOP] && prev == state_none) {
			toks->push_back(new operand <T> (-1));
			toks->push_back(&defaults <T>
				::opers[defaults <T> ::MULTOP]);
			continue;
		}
		
		// Assuming operand * operand,
		// where operand is either an
		// actual operand (value), a
		// variable, or a parenthesized
		// sub expression
		if (curr == state_operand || curr == state_operand) {
			if (prev == state_operand) {
				toks->push_back(&defaults <T>
					::opers[defaults <T> ::MULTOP]);
			}
		}

		if (curr == state_parenthesis) {
			auto ptoks = (dynamic_cast <group *> (pr.first))->toks;
			toks->push_back(new lparen);
			for (token *tptr : ptoks)
				toks->push_back(tptr);
			toks->push_back(new rparen);
		} else {
			toks->push_back(pr.first);
		}

		// change size_t to some
		// larger data holder
		if (index == (size_t) -1)
			break;
	}

	dp_msg("TOKENS");
	for (token *t : *toks)
		dp_var(t->str());

	return *toks;
}

* template <class T>
const std::pair <token *, size_t> &expression <T> ::next
	(const std::string &str, const size_t &index,
	m_state &state, var_stack <T> &vst)
{
	auto null = defaults <T> ::opers[defaults <T> ::NOPERS];
	auto opers = defaults <T> ::opers;

	std::vector <token *> ptokens;

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
					ptokens = symbols(paren, vst);

					* node *ptree = build(paren, m_params, m_map);

					pr->first = new group(ptree);
					// print(ptree, 0, 0);
					pr->second = j + 1; *				pr->first = new group(ptokens);
					pr->second = j + 1;
					state = state_operand;
					break;
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
		variable <T> var;

		try {
			if (vst.empty())
				throw not_found_exception();

			var = vst.find(accum);
			pr->first = new operand <T> (var.get());
			pr->second = i + 1;
			state = state_operand;
			break;
		} catch (...) {}

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
} */

#endif
