#ifndef LEXER_H_
#define LEXER_H_

// Engine headers
#include "feeder.hpp"
#include "ltag.hpp"

namespace zhetapi {

// Lexer
class Lexer {
	size_t 		_line	= 1;
	char   		_next	= ' ';
	Feeder *	_fd	= nullptr;
public:
	Lexer(Feeder *);

	static inline size_t get_code(char);
	static inline size_t get_code(const std::string &);
	
	static inline bool good_ident(char);

	inline void *check_dual(char, LexTag, LexTag) const;

	void *scan();
};

}

#endif