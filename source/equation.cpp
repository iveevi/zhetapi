#include <set>

#include <equation.hpp>
#include <function.hpp>

namespace zhetapi {

Equation::Equation(const std::vector <std::string> &exprs)
{
	// For now use an assert
	assert(exprs.size() > 1);

	// Rest
	__engine = new Engine();

	std::set <std::string> excl;
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

		for (std::string str : args)
			excl.insert(excl.begin(), str);
	}

	for (std::string str : excl)
		__args.push_back(str);
}

// Properties
size_t Equation::args() const
{
	return __args.size();
}

// Methods
Solutions Equation::solve() const
{
	// TODO: For now only 1 variable eqs
	if (args() != 1)
		return {{nullptr}};

	using namespace std;
	cout << "dealing with first two equations only..." << endl;

	node_manager nm = __expressions[0] - __expressions[1];

	Function f("f", __args, nm);
	Function df = f.differentiate("x");

	// Solve with a hybrid of Newton-Raphsons and Bisection
	// This assumes that obj is a continous function
	cout << f << endl;
	cout << df << endl;

	// Epsilon (accuracy)
	Token *eps = new opd_r(1e-10);
	
	// Initial guess (update heurestic)
	Token *x0 = new opd_z(1);

	Token *ft;
	Token *dft;
	Token *qt;
	Token *st;

	// Make a singleton of this
	Token *true_tok = new opd_b(true);

	// Set maximum iteration through analysis later
	const size_t MAX_ITERS = 1000;

	for (size_t i = 0; i < MAX_ITERS; i++) {
		cout << "x0 @ " << x0 << endl;
		cout << "\t = " << x0->str() << endl;
		
		// FIXME: Why cant we pass {x0}?
		ft = f({x0->copy()});

		st = __engine->compute(">", {eps, ft});
		if (tokcmp(st, true_tok))
			break;

		dft = df({x0->copy()});

		qt = __engine->compute("/", {ft, dft});
		x0 = __engine->compute("-", {x0, qt});
	}

	return {{x0}};
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

std::ostream &operator<<(std::ostream &os, const Equation &eq)
{
	os << eq.representation();

	return os;
}

}
