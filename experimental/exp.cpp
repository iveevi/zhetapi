#include <engine.hpp>

#include <lang/compilation.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

void diagnose(const node &nd)
{
	// check for monotone conditions
}

int main()
{
	// Setup
	Engine *eng = new Engine;

	const std::string code = R"(
	x = 0
	i = 0

	while (i++ <= 10)
		x = x + i * i
	)";

	Args args;
	Pardon pardon;

	// Compiling
	node_manager nm = lang::compile_block(eng, code, args, pardon);

	nm.print();

	nm[2].print();

	nm.sequential_value(eng);

	eng->list();

	diagnose(nd);
}