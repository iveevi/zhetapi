#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

/* zhetapi::StringFeeder feeder1(R"(
if (false)
	println("TRUE")
println("OUT OF IF")
)"); */

// TODO: keywords should not be separated by spaces (account for whitespace)
// TODO: test evaluation of functions with blanks (and clean up the node_value
// function by using node_functor)
zhetapi::StringFeeder feeder(R"(
alg foo(x, y, z) {
	println("x = ", x)
}
)");

int main()
{
	Engine *context = new Engine(true);

	Pardon pardon;

	// node_manager nm = cc_parse(&feeder, context, {}, pardon);

	// nm.print();

	// nm.write(cout);

	parse_global(&feeder, context);

	context->list();
}
