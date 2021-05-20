#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

StringFeeder feeder(R"(
println("Hello world - str = \"yo\"")
x = 0
y = 0, z = 12
func(12, 13, 14, 15)
ftr(12, 13
	45,
	46, 67)
ftnl(12,34
	23, 23
)
)");

int main()
{
	Engine *context = new Engine(true);

	parse_global(&feeder, context);
}
