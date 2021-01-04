#include "global.hpp"

zhetapi::Token *print(const std::vector <zhetapi::Token *> &ins)
{
	for (zhetapi::Token *tptr : ins)
		cout << tptr->str();
	
	return ins[0];
}

zhetapi::Token *println(const std::vector <zhetapi::Token *> &ins)
{
	for (zhetapi::Token *tptr : ins)
		cout << tptr->str();
	
	cout << "\n";
	
	return ins[0];
}
