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
	node_manager nm(ctx, "3 / x^2", {"x"});

	std::cout << "Compiled tree:" << std::endl;
	nm.print();
}
