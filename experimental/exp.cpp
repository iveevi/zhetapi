#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

/* zhetapi::StringFeeder feeder1(R"(
// f(x, y = 907
z = f(x, y) = x^2 + y^2
g(x, y, z) = f(x, y) = x^2 + y^2
// g(x) = f(x, y) = x^2 + y^2
// px = 127
// px0 = 127
// 0px = 54
$px = 127
println("px = ", 27)
)"); */

zhetapi::StringFeeder feeder(R"(
px = 0
/* while (px < 10) {
	println("px = ", px)

	px = px + 1
}
println("px = ", px) */
)");

int main()
{
	Engine *context = new Engine(true);

	Pardon pardon;

	node_manager nm = cc_parse(&feeder, context, {}, pardon);

	nm.print();

	nm.write(cout);
}
