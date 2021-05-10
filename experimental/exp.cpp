#include <engine.hpp>

#include <lang/compilation.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

// For interval construction
bool is_relative_comparison(code c)
{
	return (c == xge)
		|| (c == xle)
		|| (c == geq)
		|| (c == leq);
}

bool is_disjoint_monotone(const node &nd)
{
	operation_holder *ophptr = nd.cast <operation_holder> ();

	if (ophptr);
}

void diagnose(const node &nd)
{
	// check for monotone conditions
	// granted that nd is while
	node condition = nd[0];

	cout << "condition:" << endl;
	condition.print();
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

	diagnose(nm[2]);

	// GNN
	NetNode <double> n;
}