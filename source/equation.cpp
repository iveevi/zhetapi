#include <equation.hpp>

namespace zhetapi {

Equation::Equation(const std::vector <std::string> &exprs)
{
	__engine = new Engine();

	for (const std::string &str : exprs) {
		std::vector <std::string> args;
		
		while (true) {
			try {
				node_manager nm(str, args, __engine);

				__expressions.push_back(nm);
			} catch (const node_manager::undefined_symbol &e) {
				args.push_back(e.what());
			}
		}
	}
}

}