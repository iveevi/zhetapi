#include <engine.hpp>
#include <lang/error_handling.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	cout << levenshtein("sitting", "kitten") << endl;

	Engine *engine = new Engine(true);

	cout << "syms:" << endl;
	for (auto str : engine->symbol_list())
		cout << "\t" << str << endl;
	
	cout << "suggestions:" << endl;
	for (auto str : symbol_suggestions("nill", engine->symbol_list()))
		cout << "\t" << str << endl;
	
	symbol_error_msg("nill", engine);
}
