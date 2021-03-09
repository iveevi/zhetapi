#include <iostream>

#include <barn.hpp>

using namespace zhetapi;
using namespace std;

ZHETAPI_REGISTER(bt_print)
{
	for (Token *tptr : inputs)
		cout << tptr->str();
	
	return nullptr;
}

ZHETAPI_REGISTER(bt_println)
{
	for (Token *tptr : inputs)
		cout << tptr->str();
	
	cout << "\n";
	
	return nullptr;
}
