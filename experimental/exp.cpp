#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

zhetapi::StringFeeder feeder1(R"(
// px = 0
println("px = ", 27)
)");

zhetapi::StringFeeder feeder2(R"(
px = 0
while (px < 10) {
	println("px = ", px)

	px = px + 1
}
)");

int main()
{
	Engine *context = new Engine(true);

	cc_parse(&feeder1, context).print();

	parse_global(&feeder2, context);
}
