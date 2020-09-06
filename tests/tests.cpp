// Engine headers
#include <node_manager.hpp>
#include <barn.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	std::string str;

	cout << endl << "Beginning Tests..." << endl;

	while (getline(cin, str)) {
		zhetapi::node_manager <double, int> tmp(str);

		zhetapi::token *tptr = tmp.value();

		cout << endl << "Value: " << tptr->str() << " (" << tptr << ")" << endl;
	}

	return 0;
}
