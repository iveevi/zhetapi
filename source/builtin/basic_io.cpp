#include <iostream>

#include <barn.hpp>

using namespace zhetapi;
using namespace std;

ZHETAPI_REGISTER(print)
{
	for (Token *tptr : inputs)
		cout << tptr->str();
	
	return nullptr;
}

ZHETAPI_REGISTER(println)
{
	for (Token *tptr : inputs)
		cout << tptr->str();
	
	cout << "\n";
	
	return nullptr;
}
