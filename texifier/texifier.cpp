#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#include "../build/texifier.tab.c"
#include "../build/texifier.yy.c"

int main()
{
	ofstream fout("../build/texifier.out");
	ifstream fin("../build/texifier.in");
	
	string *output;
	string input;

	getline(fin, input);

#ifdef DEBUG
	cout << "[Texifier]: received \"" << input << "\"" << endl;
#endif

	char *cpy = new char[input.length() + 1];
	int i;
	
	for (i = 0; i < input.length(); i++)
		cpy[i] = input[i];
	cpy[i] = '\n';

	output = new string();
	
	yy_scan_string(cpy);
	yyparse(output);

	fout << *output << endl;

#ifdef DEBUG
	cout << "[Texifier]: returning \"" << *output << "\"" << endl;
#endif

	return 0;
}
