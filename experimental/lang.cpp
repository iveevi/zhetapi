#include <iostream>
#include <queue>
#include <mutex>
#include <vector>

#include "../engine/core/enode.hpp"
#include "../engine/core/object.hpp"
#include "../engine/lang/feeder.hpp"
#include "../engine/ads/tsqueue.hpp"
#include "../engine/core/variant.hpp"

using namespace std;
using namespace zhetapi;

// Lexer tag enumerations
enum LexTag : size_t {
	DONE, LOGIC_AND, BIT_AND,
	LOGIC_OR, BIT_OR, LOGIC_EQ,
	ASSIGN_EQ, LOGIC_NOT,
	BIT_NOT, GEQ, GE, LEQ, LE,
	LBRACE, RBRACE,
	IDENTIFIER, INTEGER
};

// String codes for enumerations
std::string strlex[] = {
	"DONE", "LOGIC AND", "BIT AND",
	"LOGIC OR", "BIT OR", "LOGIC EQ",
	"ASSIGN EQ", "LOGIC NOT",
	"BIT NOT", "GEQ", "GE", "LEQ", "LE",
	"LBRACE", "RBRACE",
	"IDENTIFIER", "INTEGER"
};

// Tokens
struct Normal {
	size_t id;
};

struct Identifier {
	size_t id = IDENTIFIER;
	std::string ident;

	Identifier(const std::string str) : ident(str) {}
};

struct Integer {
	size_t id = INTEGER;
	int value;

	Integer(int i) : value(i) {}
};

inline constexpr LexTag get_ltag(void *ltag)
{
	return *((LexTag *) ltag);
}

void free_ltag(void *ltag)
{
	LexTag id = *((LexTag *) ltag);
	if (id == IDENTIFIER)
		delete (Identifier *) ltag;
	else if (id == INTEGER)
		delete (Integer *) ltag;
	else
		delete (Normal *) ltag;	
}

// Lexer
class Lexer {
	size_t 		_line	= 1;
	char   		_next	= ' ';
	Feeder *	_fd	= nullptr;
public:
	Lexer(Feeder *fd) : _fd(fd) {}

	inline void *check_dual(char expect, LexTag succcess, LexTag fail) const {
		if (_fd->peek() == expect) {
			_fd->feed();

			return new Normal {succcess};
		}

		return new Normal {fail};
	}

	inline size_t get_code(char c) const {
		static const std::unordered_map <char, LexTag> ltags {
			{'{', LBRACE},
			{'}', RBRACE},
			{-1, DONE}
		};

		if (ltags.find(c) != ltags.end())
			return ltags.at(c);
		
		// Return offset character to avoid overlap
		// TODO: fix a constant offset
		cout << "c = " << (int) c << endl;
		return (size_t) c + 128;
	}

	void *scan() {
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
		}

		// Deal with numbers later
		if (isdigit(_next)) {
			while (isdigit((_next = _fd->feed())));
		}
		
		// Identifier
		if ((_next == '_') || isalpha(_next)) {
			std::string ident(1, _next);
			while (!isspace((_next = _fd->feed())))
				ident += _next;
			
			// cout << "<ident>\"" << ident << "\"</ident>\n";

			return (void *) (new Identifier(ident));
		}

		// Return normal symbol according to character-code table
		return new Normal {get_code(_next)};
	}
};

// Parser class
class Parser {
	// TODO: should take the tsqueue, not the lexer itself
	Lexer *			_lexer = nullptr;

	// Symbol table: string to index
	Strtable <size_t>	_hash;

	// Symbol table: index to value
	std::vector <Variant>	_vregs;
public:
	Parser(Lexer *lexer) : _lexer(lexer) {}

	void require(LexTag ltag) const {
		void *ptr = _lexer->scan();

		if (get_ltag(ptr) != ltag)
			cout << "Did not match requirements..." << endl;
		else
			cout << "Matched requirements..." << endl;
	}

	void run() const {
		require(IDENTIFIER);
		require(ASSIGN_EQ);
		require(IDENTIFIER);
	}
};

// Sources
StringFeeder sf1(R"(
==
myvar34
>=
myvar myvar4690
!mvar1

{
	thisisavar
}
)");

StringFeeder sf2 = sf1;

StringFeeder sf3(R"(
myvar1 = myvar2s
)");

// Lexers
Lexer lexer1(&sf2);
Lexer lexer2(&sf3);

// Parsers
Parser parser(&lexer2);

int main()
{
	/* Object size test
	cout << "sizeof Object = " << sizeof(Object) << endl;
	cout << "sizeof Enode = " << sizeof(Enode) << endl;
	cout << "sizeof Enode::Data = " << sizeof(Enode::Data) << endl; */

	// Lexer test
	cout << "Contents of sf1: \"";
	while (!sf1.done())
		cout << sf1.feed();
	cout << "\"" << endl;

	void *ltag;
	
	// queue <void *> tags;
	ads::TSQueue <void *> tags;

	cout << "Pushing tags:" << endl;
	while ((ltag = lexer1.scan()) != (void *) DONE) {
		cout << "\tLexTag: " << ltag << " -> " << strlex[*((LexTag *) ltag)] << endl;
		tags.push(ltag);
	}

	while (!tags.empty()) {
		ltag = tags.pop();

		free_ltag(ltag);
	}

	// Parser test
	parser.run();

	/* Object tests
	Object str = mk_str("hello world!");
	str.debug();

	Object arr[4] {
		str,
		mk_str("one"),
		mk_str("four"),
		mk_str("three hundred")
	};

	Object col = mk_col(arr, 4);
	col.debug(); */
}
