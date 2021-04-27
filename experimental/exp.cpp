#include <equation.hpp>

using namespace std;
using namespace zhetapi;

/*
bool var_pred(lbl label)
{
	return (label == l_variable);
}

void fit(const node &a, const node &fitter)
{
	if (tokcmp(a.ptr(), fitter.ptr())) {
		cout << "Matched " << fitter.ptr()->str() << endl;
	}

	// Use this approach for commutative operations
	if (!is_commutative(fitter)) {
		cout << "Non commutative comparison, skip for now" << endl;

		return;
	}

	vector <node> not_wcs;

	// Add a begin and end method for nodes
	for (node nd : fitter._leaves) {
		if (nd.caller() != Token::token_wildcard)
			not_wcs.push_back(nd);
	}

	// Check the rest of the nodes
	vector <node> current = a._leaves;

	cout << "Current - before reducing:" << endl;
	for (node nd : current)
		nd.print();

	for (node rule : not_wcs) {
		vector <node> unresolved;

		for (node nd : current) {
			if (!node::loose_match(rule, nd))
				unresolved.push_back(nd);
		}

		current = unresolved;
	}

	cout << "Current - after reducing:" << endl;
	for (node nd : current)
		nd.print();
} */

int main()
{
	/* Engine *engine = new Engine();

	engine->put("x", new wildcard("x", var_pred));

	node_manager nm("2x", engine);
	node_manager nm2("2y", {"y"}, engine);

	nm.unpack();
	nm.print();

	cout << "nm2:" << endl;
	nm2.print();

	fit(nm2.get_tree(), nm.get_tree());

	Engine *engine = new Engine();

	Function f = "f(x) = x^2";

	f.print();

	engine->put(f);

	node_manager nm("df/dx", {}, engine);

	cout << "nm:" << endl;
	nm.print();

	cout << "display: " << nm.display() << endl;

	Equation eq({
		"x^2 - x",
		"6"
	});

	eq.solve(); */

	Engine *context = new Engine();
}
