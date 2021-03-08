#ifndef ENGINE_BASE_H_
#define ENGINE_BASE_H_

// C/C++ headers
#include <stdexcept>

// Engine headers
#include <token.hpp>

namespace zhetapi {

// Symbol table alias
template <class T>
using symtab = std::unordered_map <std::string, T>;

// Other aliases
using signature = std::vector <std::type_index>;
using overloads = std::vector <std::pair <signature, Token *>>;

// Class for storing overloads
class engine_base {
	symtab <overloads>	__overloads;

	// Private methods
	std::string gen_overload_msg(const signature &);
	signature gen_signature(const std::vector <Token *> &);
public:
	engine_base();

	Token *compute(const std::string &, const std::vector <Token *> &);

	std::string overload_catalog(const std::string &) const;
	
	// Exceptions
	class unknown_operation_overload : public std::runtime_error {
	public:
		explicit unknown_op_overload(const std::string &str)
				: std::runtime_error(str) {}
	};
};

}

#endif
