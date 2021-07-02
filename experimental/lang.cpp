#include "../engine/core/node_manager.hpp"
#include "../engine/core/enode.hpp"
#include "../engine/core/engine_base.hpp"
#include "../engine/engine.hpp"
#include "../engine/timer.hpp"
#include "../engine/core/enode_parser.hpp"

using namespace std;
using namespace zhetapi;

Engine *ctx = new Engine();

int main()
{
	node_manager nm(ctx, "2 / 7");

	nm.print();

	/* std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	nm = node_manager(ctx, "{3, 5, 6}", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	nm = node_manager(ctx, "\"fdfsdffsf\\t\"", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	nm = node_manager(ctx, "3/x^2", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	nm = node_manager(ctx, "3x^2", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	nm = node_manager(ctx, "x^2 + 10^2", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */
}
