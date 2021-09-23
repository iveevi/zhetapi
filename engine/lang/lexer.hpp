#ifndef LEXER_H_
#define LEXER_H_

// Standard headers
#include <string>

// Engine headers
#include "ltag.hpp"

namespace zhetapi {

// Lexer
class Lexer {
	size_t 		_line	= 1;
	size_t		_index	= 0;
	char   		_next	= ' ';
	std::string	_source;
public:
	Lexer(const std::string &str);

	// Feed functions
	inline char feed();
	inline char peek();
	inline bool done();
	inline void backup(size_t);

	// Code functions
	static inline size_t get_code(char);
	static inline size_t get_code(const std::string &);

	static inline bool good_ident(char);

	inline void *check_dual(char, LexTag, LexTag);

	// Main routine
	void *scan();
};

}

#endif
