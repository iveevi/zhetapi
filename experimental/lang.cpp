#include <iostream>

#include "../engine/core/object.hpp"

using namespace std;
using namespace zhetapi;

int main()
{
	Object str = mk_str("hello world!");

	cout << "str = \"" << (const char *) str.data << "\"" << endl;
	cout << "id = " << str.id << endl;

	str.debug();
}
