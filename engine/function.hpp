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

		static Barn <T, U> barn;
	};

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

		// Split string into expression and symbols
		for (i = 0; i < str.length(); i++) {
			if (str[i] == '=') {
				index = i;
				break;
			}
		}

		__symbol = str.substr(0, index);


		// Determine parameters' symbols
		for (start = -1, i = 0; i < __symbol.length(); i++) {
			if (str[i] == '(' && start == -1)
				start = i;
			if (str[i] == ')')
				end = i;
		}

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
}

#endif
