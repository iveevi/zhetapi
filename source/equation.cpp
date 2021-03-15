#include <equation.hpp>

namespace zhetapi {

Equation::Equation(const std::vector <std::string> &exprs)
{
	__engine = new Engine();

	for (const std::string &str : exprs) {
		std::vector <std::string> args;
		
		bool success = false;
		do {
			try {
				node_manager nm(str, args, __engine);

				__expressions.push_back(nm);

				success = true;
			} catch (const node_manager::undefined_symbol &e) {
				args.push_back(e.what());
			}
		} while(!success);
	}
}

std::string Equation::representation() const
{
	std::string out = "";

	size_t n = __expressions.size();
	for (size_t i = 0; i < n; i++) {
		out += __expressions[i].display();

		if (i < n - 1)
			out += " = ";
	}

	return out;
}

}
