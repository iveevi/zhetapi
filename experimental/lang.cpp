#include <iostream>

#include "../engine/core/enode.hpp"
#include "../engine/core/object.hpp"
#include "../engine/lang/feeder.hpp"

using namespace std;
using namespace zhetapi;

// Lexer
class Lexer {
	size_t 		_line	= 1;
	char   		_next	= ' ';
	Feeder *	_fd	= nullptr;
public:

};

int main()
{
	cout << "sizeof Object = " << sizeof(Object) << endl;
	cout << "sizeof Enode = " << sizeof(Enode) << endl;
	cout << "sizeof Enode::Data = " << sizeof(Enode::Data) << endl;

	// Lexer test

	// Object tests
	Object str = mk_str("hello world!");
	str.debug();

	Object arr[4] {
		str,
		mk_str("one"),
		mk_str("four"),
		mk_str("three hundred")
	};

	Object col = mk_col(arr, 4);
	col.debug();
}
