#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C/C++ headers
#include <string>

// Engine headers
#include <node_manager.hpp>

namespace zhetapi {

	/*
	 * Represents a mathematical function.
	 */
	template <class T, class U>
	class Function {
		std::string			__symbol;
		std::vector <std::string>	__params;
		node_manager <T, U>		__manager;
	public:
		Function();
		Function(const char *);
		Function(const std::string &);

		template <class ... A>
		token *operator()(A ...);

		template <class ... A>
		token *derivative(const std::string &, A ...);

		void print() const;
	private:
		template <class A>
		void gather(std::vector <token *> &, A);

		template <class A, class ... B>
		void gather(std::vector <token *> &, A, B ...);

		size_t index(const std::string &) const;
	public:
		// Exception classes
		class invalid_definition {};

		// Static variables
		static Barn <T, U> barn;

		static T h;
	};

	// Static
	template <class T, class U>
	Barn <T, U> Function <T, U> ::barn = Barn <T, U> ();

	template <class T, class U>
	T Function <T, U> ::h = 0.0001;

	// Constructors
	template <class T, class U>
	Function <T, U> ::Function() {}

	template <class T, class U>
	Function <T, U> ::Function(const char *str) : Function(std::string
			(str)) {}

	template <class T, class U>
	Function <T, U> ::Function(const std::string &str)
	{
		std::string pack;
		std::string tmp;

		size_t index;
		size_t start;
		size_t end;
		size_t i;

		bool valid;
		bool sb;
		bool eb;

		valid = false;
		sb = false;
		eb = false;

		// Split string into expression and symbols
		for (i = 0; i < str.length(); i++) {
			if (str[i] == '=') {
				valid = true;
				index = i;
				break;
			}
		}

		if (!valid)
			throw invalid_definition();

		__symbol = str.substr(0, index);


		// Determine parameters' symbols
		for (start = -1, i = 0; i < __symbol.length(); i++) {
			if (str[i] == '(' && start == -1) {
				start = i;
				sb = true;
			}

			if (str[i] == ')') {
				end = i;
				eb = true;
			}
		}

		if (!sb || !eb)
			throw invalid_definition();

		pack = __symbol.substr(start + 1, end - start - 1);

		for (i = 0; i < pack.length(); i++) {
			if (pack[i] == ',' && !tmp.empty()) {
				__params.push_back(tmp);
				
				tmp.clear();
			} else if (!isspace(pack[i])) {
				tmp += pack[i];
			}
		}

		if (!tmp.empty())
			__params.push_back(tmp);
		
		// Determine function's symbol
		__symbol = __symbol.substr(0, start);

		// Construct the tree manager
		__manager = node_manager <T, U> (str.substr(++index), __params);
	}

	// Computational utilities
	template <class T, class U>
	template <class ... A>
	token *Function <T, U> ::operator()(A ... args)
	{
		std::vector <token *> tokens;

		gather(tokens, args...);

		assert(tokens.size() == __params.size());

		return __manager.substitute_and_compute(tokens);
	}

	template <class T, class U>
	template <class ... A>
	token *Function <T, U> ::derivative(const std::string &str, A ... args)
	{
		std::vector <token *> tokens;

		gather(tokens, args...);

		assert(tokens.size() == __params.size());

		size_t i = index(str);

		assert(i != -1);

		// Right
		token *right;

		tokens[i] = barn.compute("+", {tokens[i], new operand <T> (h)});

		for (size_t k = 0; k < tokens.size(); k++) {
			if (k != i)
				tokens[k] = tokens[k]->copy();
		}
		
		right = __manager.substitute_and_compute(tokens);
		
		// Left
		token *left;

		tokens[i] = barn.compute("-", {tokens[i], new operand <T> (T(2) * h)});

		for (size_t k = 0; k < tokens.size(); k++) {
			if (k != i)
				tokens[k] = tokens[k]->copy();
		}

		left = __manager.substitute_and_compute(tokens);

		// Compute
		token *diff = barn.compute("-", {right, left});

		diff = barn.compute("/", {diff, new operand <T> (T(2) * h)});

		return diff;
	}

	// Printing utilities
	template <class T, class U>
	void Function <T, U> ::print() const
	{
		__manager.print();
	}

	// Gathering facilities
	template <class T, class U>
	template <class A>
	void Function <T, U> ::gather(std::vector <token *> &tokens, A in)
	{
		tokens.push_back(new operand <A>(in));
	}
	
	template <class T, class U>
	template <class A, class ... B>
	void Function <T, U> ::gather(std::vector <token *> &tokens, A in, B ... args)
	{
		tokens.push_back(new operand <A>(in));

		gather(tokens, args...);
	}

	template <class T, class U>
	size_t Function <T, U> ::index(const std::string &str) const
	{
		auto itr = std::find(__params.begin(), __params.end(), str);

		if (itr == __params.end())
			return -1;

		return std::distance(__params.begin(), itr);
	}

}

#endif
