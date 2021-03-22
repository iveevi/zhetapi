#include <engine.hpp>

#include <core/wildcard.hpp>

using namespace std;
using namespace zhetapi;

bool var_pred(lbl label)
{
	return (label == l_variable);
}

void fit(const node &a, const node &fitter)
{
	if (tokcmp(a.ptr(), fitter.ptr())) {
		cout << "Matched " << fitter.ptr()->str() << endl;
	}
}

int main()
{
	Engine *engine = new Engine();

	engine->put("x", new wildcard("x", var_pred));

	node_manager nm("x^2", engine);
	node_manager nm2("y^2", {"y"}, engine);

	nm.unpack();
	nm.print();

	cout << "nm2:" << endl;
	nm2.print();

	fit(nm2.get_tree(), nm.get_tree());
}
