#include <iostream>

#include "../engine/core/enode.hpp"
#include "../engine/core/object.hpp"
#include "../engine/lang/feeder.hpp"

using namespace std;
using namespace zhetapi;

// Lexer tag enumerations
enum LexTag : size_t {
	DONE,
	LOGIC_AND,
	BIT_AND,
	LOGIC_OR,
	BIT_OR,
	LOGIC_EQ,
	ASSIGN_EQ,
	LOGIC_NOT,
	BIT_NOT,
	GEQ,
	GE,
	LEQ,
	LE
};

// Lexer
class Lexer {
	size_t 		_line	= 1;
	char   		_next	= ' ';
	Feeder *	_fd	= nullptr;
public:
	Lexer(Feeder *fd) : _fd(fd) {}

	inline LexTag check_dual(char expect, LexTag succcess, LexTag fail) {
		if (_fd->peek() == expect) {
			_fd->feed();

			return succcess;
		}

		return fail;
	}

	// TODO: change return to "LexTag" type
	size_t scan() {
		if (_fd->done())
			return DONE;
		
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
		std::string ident(1, _next);
		while (!isspace((_next = _fd->feed())))
			ident += _next;
		
		cout << "<ident>\"" << ident << "\"</ident>\n";

		return 20;
	}
};

// Sources
StringFeeder sf1(R"(
==
myvar34
>=
myvar
)");

StringFeeder sf2 = sf1;

// Lexers
Lexer lexer(&sf2);

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

	cout << "\nLexer test:" << endl;

	size_t i;
	while ((i = lexer.scan()) != DONE)
		cout << "\tLexTag: " << i << endl;

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
