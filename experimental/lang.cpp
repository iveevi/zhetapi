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
	node_manager nm;

	/* nm = node_manager(ctx, "2 / 7");

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();
	
	nm = node_manager(ctx, "2.987 / 7 + 59.45");

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

	nm = node_manager(ctx, "25.0 + 4 == 4", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	/* nm = node_manager(ctx, "\"fdfsdffsf\\t\"", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

	/* nm = node_manager(ctx, "3/x^2", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

	/* nm = node_manager(ctx, "3x^2", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	nm = node_manager(ctx, "x^2 + 10^2", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

	std::cout << "value = " << nm.value(ctx)->dbg_str() << std::endl;
}
