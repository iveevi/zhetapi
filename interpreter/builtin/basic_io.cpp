#include <iostream>

#include <engine.hpp>

using namespace zhetapi;
using namespace std;

namespace zhetapi {

void cprint(Token *tptr)
{
	if (tptr)
		cout << tptr->dbg_str();
	else
		cout << "<Null>";
}

ZHETAPI_REGISTER(bt_print)
{
	for (Token *tptr : inputs)
		cprint(tptr);
	
	return nullptr;
}

ZHETAPI_REGISTER(bt_println)
{
	for (Token *tptr : inputs)
		cprint(tptr);
	
	cout << "\n";
	
	return nullptr;
}

}