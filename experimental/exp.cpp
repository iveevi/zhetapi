#include <engine.hpp>

using namespace zhetapi;
using namespace std;

node_manager equation_side(const string &str)
{
	Engine *engine = new Engine();

	std::vector <std::string> args;
	while (true) {
		try {
			node_manager nm(str, args, engine);

			return nm;
		} catch (const node_manager::undefined_symbol &e) {
			args.push_back(e.what());
		}
	}

	return node_manager();
}

int main()
{
	string str1 = "sin(x)^2 + cos(x)^2";
	string str2 = "1";

	Function ftr = "f(x) = sin(x)^2";

	cout << ftr << endl;

	Engine *engine = new Engine();

	node_manager left = equation_side(str1);
	node_manager right = equation_side(str2);

	cout << "left:" << endl;
	left.print();

	cout << "right:" << endl;
	right.print();
}
