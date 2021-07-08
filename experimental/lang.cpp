#include "../engine/core/node_manager.hpp"
#include "../engine/core/enode.hpp"
#include "../engine/core/engine_base.hpp"
#include "../engine/engine.hpp"
#include "../engine/timer.hpp"
#include "../engine/core/enode_parser.hpp"
#include "../engine/lang/parser.hpp"

using namespace std;
using namespace zhetapi;

Engine *ctx = new Engine();

Pardon pardon;

int main()
{
	node_manager nm;

	// Test factorial
	// nm = node_manager(ctx, "x++", {"x"});
	// nm = node_manager(ctx, "x[10]", {"x"});
	/* nm = node_manager(ctx, "x + 3", {"x"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

	// Args veq = eq_split("x=x+3");

	// cout << "veq[0] = " << veq[0] << endl;
	// cout << "veq[1] = " << veq[1] << endl;
	nm = node_manager(ctx, "x^T", {"x"}); // cc_run("x=x+3", ctx, {"x"}, pardon);

	std::cout << "Tree:" << std::endl;
	nm.print();

	/* nm = node_manager(ctx, "12 * index", {"dex", "index"}); // cc_run("x=x+3", ctx, {"x"}, pardon);

	std::cout << "Tree:" << std::endl;
	nm.print();
	
	nm = node_manager(ctx, "12 in dex", {"dex", "index"}); // cc_run("x=x+3", ctx, {"x"}, pardon);

	std::cout << "Tree:" << std::endl;
	nm.print(); */

	/* nm = node_manager(ctx, "[[2 + 54, 2 * 76], [90 * 9, 60]]");

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

	/* nm = node_manager(ctx, "2.987 / 7 + 59.45");

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

	/* nm = node_manager(ctx, "{}");

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print();

	nm = node_manager(ctx, "12 index", {"dex"});

	std::cout << "==============" << std::endl;
	std::cout << "Compiled tree:" << std::endl;
	nm.print(); */

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

	// cout << "value = " << nm.value(ctx)->dbg_str() << std::endl;
}
