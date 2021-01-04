#include "global.hpp"

Token *print(const vector <Token *> &ins)
{
	for (Token *tptr : ins)
		cout << tptr->str();
	
	return ins[0];
}

Token *println(const vector <Token *> &ins)
{
	for (Token *tptr : ins)
		cout << tptr->str();
	
	cout << "\n";
	
	return ins[0];
}
