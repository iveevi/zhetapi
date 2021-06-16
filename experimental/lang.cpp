#include "../engine/lang/parser.hpp"

// Namespaces
using namespace std;
using namespace zhetapi;

// TODO: keywords should not be separated by spaces (account for whitespace)
// TODO: test evaluation of functions with blanks (and clean up the node_value
// TODO: what about an assignment or statement before a comment? (eg. x = 0 // some comment...)
// function by using node_functor)
// TODO: allow global to be multiple lines
zhetapi::StringFeeder feeder(R"(
alg foo() {
	// println("x = ", x)
	println("x = ...")
	return 46
}

global x = 65
global str = "hello world"
global y = 76, z = 80
)");

zhetapi::StringFeeder feeder2(R"(
test.foo()
println("test.str = ", test.str)
println("sum of x, y, z = ", test.x + test.y + test.z)

global main_glob = 7756.3243

println("main glob = ", main_glob)
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

	// TODO: would need to enclose this is a pushed stack (to avoid name issues)
	Engine *nctx = push_and_ret_stack(context);

	mdl_parse(&feeder, nctx, module);

	// TODO: add a list method
	// cout << "module = " << module->dbg_str() << endl;

	context->put("test", module);

	// module->list_attributes(cout);
	// context->list();

	cout << "PARSING MAIN CODE:" << endl;
	parse_global(&feeder2, context);
}
