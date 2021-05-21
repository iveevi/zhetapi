#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

zhetapi::StringFeeder feeder(R"(
px = 10
f(x, y) = x^2 + y^2
println("px = ", px)
println("f @ (3, 4) = ", f(3, 4))
)");

int main()
{
	Engine *context = new Engine(true);

	parse_global(&feeder, context);
}
