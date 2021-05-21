#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

StringFeeder feeder(R"(
if (false) {
	println("YES")
} elif (4 == 4) {
	println("ELIF!")
} else {
	println("ELSE!")
}
)");

int main()
{
	Engine *context = new Engine(true);

	parse_global(&feeder, context);
}
