#include <set>

#include <equation.hpp>
#include <function.hpp>

namespace zhetapi {

// Put in another file
const std::string netwon_algorithm = R"(
less = 0
more = 0

fp = f.derivative()

println("LESS: ", less)
println("r: ", r)
println("FUNC: ", f)
println("FUNC-p: ", fp)
)";

algorithm newton_full("newton_full", netwon_algorithm, {"r", "f"});

Equation::Equation(const std::vector <std::string> &exprs)
{
	if (exprs.size() <= 1)
		throw bad_input_size();

	// Rest
	_engine = new Engine();

	if (newton_full.empty())
		newton_full.compile(_engine);
	
	newton_full.print();

	std::set <std::string> excl;
	for (const std::string &str : exprs) {
		std::vector <std::string> args;
		
		bool success = false;
		do {
			try {
				node_manager nm(_engine, str, args);

				_expressions.push_back(nm);

				success = true;
			} catch (const node_manager::undefined_symbol &e) {
				args.push_back(e.what());
			}
		} while(!success);

		for (std::string str : args)
			excl.insert(excl.begin(), str);
	}

	for (std::string str : excl)
		_args.push_back(str);
}

// Properties
size_t Equation::args() const
{
	return _args.size();
}

// Methods
Solutions Equation::solve() const
{
	// TODO: For now only 1 variable eqs
	if (args() != 1)
		return {{nullptr}};

	using namespace std;
	cout << "dealing with first two equations only..." << endl;

	node_manager nm = _expressions[0] - _expressions[1];

	Function f("f", _args, nm);
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

	newton_full.execute(_engine, {x0, f.copy()});

	// Make a singleton of this
	Token *true_tok = new opd_b(true);

	// Set maximum iteration through analysis later
	const size_t MAX_ITERS = 1000;

	for (size_t i = 0; i < MAX_ITERS; i++) {
		cout << "x0 @ " << x0 << endl;
		cout << "\t = " << x0->dbg_str() << endl;
		
		// FIXME: Why cant we pass {x0}?
		ft = f({x0->copy()});

		st = _engine->compute(">", {eps, ft});
		if (tokcmp(st, true_tok))
			break;

		dft = df({x0->copy()});

		qt = _engine->compute("/", {ft, dft});
		x0 = _engine->compute("-", {x0, qt});
	}

	return {{x0}};
}

std::string Equation::representation() const
{
	std::string out = "";

	size_t n = _expressions.size();
	for (size_t i = 0; i < n; i++) {
		out += _expressions[i].display();

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
