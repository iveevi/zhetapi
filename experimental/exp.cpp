#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

/* println("Hello world - str = \"yo\"")
x = 12
y = 0, z = 12
func(12, 13, 14, 15)
ftr(12, 13
	45,
	46, 67)
ftnl(12,34
	23, 23
)
println("x = ", x) */

StringFeeder feeder(R"(
if (true) {
	println("Hi!")
}

println("Post first if")
if (true)
	println("Hi!")

println("Post second if")
if (false) {
	println("SHOULDNT BE HERE")
}

println("Post third if")
if (false)
	println("SHOULDNT BE HERE")

println("Post fourth if")
)");

int main()
{
	Engine *context = new Engine(true);

	parse_global(&feeder, context);
}
