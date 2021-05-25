#include <lang/parser.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

/* zhetapi::StringFeeder feeder1(R"(
if (false)
	println("TRUE")
println("OUT OF IF")
)"); */

zhetapi::StringFeeder feeder(R"(
px = 0
if (true)
	println("IF!")
elif (px > 0) {
	println("ELIF!")
	println("second statement")
} else
	println("ELSE!")
)");

int main()
{
	Engine *context = new Engine(true);

	Pardon pardon;

	node_manager nm = cc_parse(&feeder, context, {}, pardon);

	nm.print();

	// nm.write(cout);
}
