#include <engine.hpp>

#include <core/wildcard.hpp>

using namespace std;
using namespace zhetapi;

bool var_pred(lbl label)
{
	return (label == l_variable);
}

int main()
{
	Engine *engine = new Engine();

	engine->put("x", new wildcard("x", var_pred));

	node_manager nm("x^2", engine);
	
	nm.print();
	nm.unpack();
	nm.print();
}
