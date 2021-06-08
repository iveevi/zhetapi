#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

// TODO: keywords should not be separated by spaces (account for whitespace)
// TODO: test evaluation of functions with blanks (and clean up the node_value
// TODO: what about an assignment or statement before a comment? (eg. x = 0 // some comment...)
// function by using node_functor)
zhetapi::StringFeeder feeder(R"(
alg foo() {
	// println("x = ", x)
	println("x = ...")
	return 46
}
)");

zhetapi::StringFeeder feeder2(R"(
test.foo()
)");

int main()
{
	Engine *context = new Engine(true);

	Pardon pardon;

	/* node_manager nm = cc_parse(&feeder, context, {}, pardon);
	nm.print(); */

	// nm.write(cout);

	/* int ret = parse_global(&feeder, context);
	cout << "return = " << ret << endl;
	context->list(); */

	Module *module = new Module("test");

	mld_parse(&feeder, context, module);

	// TODO: add a list method
	cout << "module = " << module->dbg_str() << endl;

	module->list_attributes(cout);

	context->put("test", module);

	context->list();

	parse_global(&feeder2, context);
}
