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
	DONE, COMMA,
	LOGIC_AND, BIT_AND,
	LOGIC_OR, BIT_OR, LOGIC_EQ,
	ASSIGN_EQ, LOGIC_NOT,
	BIT_NOT, GEQ, GE, LEQ, LE,
	LPAREN, RPAREN,
	LBRACE, RBRACE,
	IDENTIFIER, INTEGER,
	ALGORITHM
};

// String codes for enumerations
std::string strlex[] = {
	"DONE", "COMMA",
	"LOGIC AND", "BIT AND",
	"LOGIC OR", "BIT OR", "LOGIC EQ",
	"ASSIGN EQ", "LOGIC NOT",
	"BIT NOT", "GEQ", "GE", "LEQ", "LE",
	"LPAREN", "RPAREN",
	"LBRACE", "RBRACE",
	"IDENTIFIER", "INTEGER",
	"ALGORITHM"
};

// Tokens
struct Normal {
	size_t id;
};

struct Identifier {
	size_t id = IDENTIFIER;
	std::string ident;

	Identifier(const std::string str) : ident(str) {}

	// Value function
	static std::string cast(void *ptr) {
		return ((Identifier *) ptr)->ident;
	}
};

// Should generalize to primitives
struct Integer {
	size_t id = INTEGER;
	int value;

	Integer(int i) : value(i) {}

	// Value function
	static int cast(void *ptr) {
		return ((Integer *) ptr)->value;
	}
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

	static inline size_t get_code(char c) {
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
		cout << "c = " << (int) c << endl;
		return (size_t) c + 128;
	}

	static inline size_t get_code(const std::string &str) {
		static const std::unordered_map <std::string, LexTag> ltags {
			{"alg", ALGORITHM}
		};

		if (ltags.find(str) != ltags.end())
			return ltags.at(str);
		
		cout << "str = " << str << endl;
		return DONE;
	}
	
	static inline bool good_ident(char c) {
		// cout << "gic = " << c << endl;
		return !isspace(c) && (isdigit(c) || isalpha(c) || c == '_');
	}

	inline void *check_dual(char expect, LexTag succcess, LexTag fail) const {
		if (_fd->peek() == expect) {
			_fd->feed();

			return new Normal {succcess};
		}

		return new Normal {fail};
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
		case ',':
			return (void *) COMMA;
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
};

// Parser class
// TODO: should only take a tsqueue of tags,
// for parallelization
class Parser {
	// TODO: should take the tsqueue, not the lexer itself
	ads::TSQueue <void *>  *_tsq = nullptr;

	// Symbol table: string to index
	Strtable <size_t>	_hash;

	// Symbol table: index to value
	std::vector <Variant>	_vregs;

	// Private structs
	struct TagPair {
		void *data;
		LexTag tag;
	};
public:
	Parser(ads::TSQueue <void *> *queue) : _tsq(queue) {}

	TagPair require(LexTag ltag) const {
		auto pr = get();
		cout << "REQ-GOT Tag: " << strlex[pr.tag] << endl;

		if (pr.tag != ltag)
			throw bad_tag(pr.tag, ltag);
		
		return pr;
	}

	TagPair get() const {
		if (_tsq->empty())
			throw eoq();

		void *ptr = _tsq->pop();
		return {ptr, get_ltag(ptr)};
	}

	void alg() const {
		std::string ident;
		
		// TODO: need a get function that checks for empty-ness
		auto pr = require(IDENTIFIER);
		// if (pr.tag == IDENTIFIER)
		cout << "Identifier! :) -> " << Identifier::cast(pr.data) << endl;

		require(LPAREN);
		require(RPAREN);
	}

	void run() const {
		void *ptr;
		while (!_tsq->empty()) {
			ptr = _tsq->pop();

			LexTag ltag = get_ltag(ptr);

			cout << "Tag = " << strlex[ltag] << endl;
			if (ltag == ALGORITHM) {
				cout << "ALGORITHM!!" << endl;
				alg();
			}
		}
		
		cout << "Done." << endl;
	}

	// Exceptions
	class eoq : public std::runtime_error {
	public:
		eoq() : std::runtime_error("Parser: end of tag queue") {}
	};

	class bad_tag : public std::runtime_error {
	public:
		bad_tag(LexTag got, LexTag exp)
			: std::runtime_error("Parser: unexpected tag <"
				+ strlex[got] + ">, expected <"
				+ strlex[exp] + ">") {}
	};
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
alg myalg() {
	x = 21
}
)");

// Lexers
Lexer lexer1(&sf2);
Lexer lexer2(&sf3);

int main()
{
	/* Object size test
	cout << "sizeof Object = " << sizeof(Object) << endl;
	cout << "sizeof Enode = " << sizeof(Enode) << endl;
	cout << "sizeof Enode::Data = " << sizeof(Enode::Data) << endl; */

	// Lexer test
	/* cout << "Contents of sf1: \"";
	while (!sf1.done())
		cout << sf1.feed();
	cout << "\"" << endl; */

	void *ltag;
	
	// queue <void *> tags;
	ads::TSQueue <void *> tags;

	cout << "Pushing tags:" << endl;
	while ((ltag = lexer2.scan()) != (void *) DONE) {
		cout << "\tLexTag: " << ltag << " -> " << strlex[*((LexTag *) ltag)] << endl;
		tags.push(ltag);
	}

	// Parser test
	Parser parser(&tags);

	cout << "Parser-------------------------->" << endl;
	parser.run();

	// Free the elements of the queue
	while (!tags.empty()) {
		ltag = tags.pop();

		free_ltag(ltag);
	}
}
