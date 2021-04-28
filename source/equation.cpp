#include <set>

#include <equation.hpp>
#include <function.hpp>

namespace zhetapi {

// Algorithm code (put in a separate file)
const std::string alg_newton_iter = R"(
        return r - f(r)/fp(r)
)";

const std::string alg_newton = R"(
        # Should be static
	eps = 1e-10

	# Get the derivative
	fp = f.derivative()

	less = null
	more = null

	y = f(r)
	if (y > 0)
		more = r
	else
		less = r
	
	x = r

	while ((more == null) || (less == null)) {
		x = newton_iter(x, f, fp)

		y = f(x)
		if ((y > 0) && (more == null))
			more = x
		elif ((y < 0) && (less == null))
			less = x
	}

	while (true) {
		x = newton_iter(x, f, fp)
                
		if ((x < less) || (x > more))
			x = 0.5 * (more + less)

		# Check for exit condition (interval or value)
		y = f(x)
		if ((abs(y) < eps) || (abs(more - less) < eps))
			break

                # Adjust boundaries
                if (y < 0) {
                        # Make sure we are actually closing the interval
                        if (abs(more - less) > abs(more - x))
                                less = x
                } else {
                        if (abs(more - less) > abs(more - x))
                                more = x
                }
	}

        return x
)";

// Algorithm objects
algorithm A_newton("newton", alg_newton, {"r", "f"});
algorithm A_newton_iter("newton_iter", alg_newton_iter, {"r", "f", "fp"});

// TODO: Later, just parse algorithm code into Engines with parse::parse

// Equation
Equation::Equation(const std::vector <std::string> &exprs)
{
        // Size check
	if (exprs.size() <= 1)
		throw bad_input_size();

	// Initialize engine and algorithms
	_engine = new Engine(true);

	if (A_newton_iter.empty())
		A_newton_iter.compile(_engine);
        _engine->put(A_newton_iter);

	if (A_newton.empty())
		A_newton.compile(_engine);
        _engine->put(A_newton);

        // Separating the sides of the equation
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

	x0 = A_newton.execute(_engine, {x0, f.copy()});

        cout << "x0 = " << x0->dbg_str() << endl;

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
