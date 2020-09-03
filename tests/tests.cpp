// Boost headers
#include <boost/spirit/include/qi.hpp>

// Engine headers
#include <node_manager.hpp>

// #define DEBUG_PARSER 1

using namespace std;

int main()
{
	std::string str;

	while (getline(cin, str))
		zhetapi::node_manager <double, int> tmp(str);

	return 0;
}
