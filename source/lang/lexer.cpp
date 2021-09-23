#include "../../engine/lang/lexer.hpp"

namespace zhetapi {

Lexer::Lexer(Feeder *fd) : _fd(fd) {}

inline size_t Lexer::get_code(char c) {
	static const std::unordered_map <char, LexTag> ltags {
		{'{', LBRACE},
		{'}', RBRACE},
		{'(', LPAREN},
		{')', RPAREN},
		{-1, DONE}
	};

	if (ltags.find(c) != ltags.end())
		return ltags.at(c);
	
	// Return offset character to avoid overlap
	// TODO: fix a constant offset
	std::cout << "c = " << (int) c << std::endl;
	return (size_t) c + 128;
}

inline size_t Lexer::get_code(const std::string &str) {
	static const std::unordered_map <std::string, LexTag> ltags {
		{"alg", ALGORITHM}
	};

	if (ltags.find(str) != ltags.end())
		return ltags.at(str);
	
	std::cout << "str = " << str << std::endl;
	return DONE;
}

inline bool Lexer::good_ident(char c) {
	// cout << "gic = " << c << endl;
	return !isspace(c) && (isdigit(c) || isalpha(c) || c == '_');
}

inline void *Lexer::check_dual(char expect, LexTag succcess, LexTag fail) const {
	if (_fd->peek() == expect) {
		_fd->feed();

		return new Normal {succcess};
	}

	return new Normal {fail};
}

void *Lexer::scan() {
	if (_fd->done())
		return (void *) DONE;
	
	// Read under non-space character
	while ((_next = _fd->feed()) != EOF) {
		if (_next == '\n')
			_line++;
		else if (!isspace(_next))
			break;
	}

	// Check the first non-space character
	// TODO: put into another function
	switch (_next) {
	case ',':
		return new Normal {COMMA};
	case '&':
		return check_dual('&', LOGIC_AND, BIT_AND);
	case '|':
		return check_dual('|', LOGIC_OR, BIT_OR);
	case '=':
		return check_dual('=', LOGIC_EQ, ASSIGN_EQ);
	case '!':
		return check_dual('=', LOGIC_NOT, BIT_NOT);
	case '>':
		return check_dual('=', GEQ, GE);
	case '<':
		return check_dual('=', LEQ, LE);
	case '+':
		return check_dual('=', PLUS, PLUS_EQ);
	}

	// Deal with numbers later
	if (isdigit(_next)) {
		while (isdigit((_next = _fd->feed())));

		return (void *) (new Integer(1));
	}
	
	// Identifier
	if ((_next == '_') || isalpha(_next)) {
		std::string ident(1, _next);
		while (good_ident(_next = _fd->feed()))
			ident += _next;
		_fd->backup(1);
		
		// cout << "_next = \'" << _next << '\'' << endl;
		// cout << "<ident>\"" << ident << "\"</ident>\n";
		size_t pos;
		if (pos = get_code(ident))
			return new Normal {pos};

		return (void *) (new Identifier(ident));
	}

	// Return normal symbol according to character-code table
	return new Normal {get_code(_next)};
}

}