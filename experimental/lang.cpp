#include <iostream>

#include "../engine/core/object.hpp"

using namespace std;
using namespace zhetapi;

int main()
{
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
