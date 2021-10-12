#include <iostream>
#include <mutex>
#include <queue>
#include <stack>
#include <vector>

#include "../engine/ads/tsqueue.hpp"
#include "../engine/core/enode.hpp"
#include "../engine/core/object.hpp"
#include "../engine/core/variant.hpp"
#include "../engine/lang/feeder.hpp"
#include "../engine/lang/lexer.hpp"
#include "../engine/lang/ltag.hpp"
#include "../engine/lang/parser.hpp"

using namespace std;
using namespace zhetapi;

// Lexers
Lexer lexer(R"(
myvar = 21.14 + 21 * 53454 / 3
x = 10
)");

// f(x, y, z) = 2 * x + 25.423 + y - 3.0 * z
// myvar2 = 210
// println(10 + 13)

int main()
{
	void *ltag;

	// queue <void *> tags;
	ads::TSQueue <void *> tags;

	cout << "Pushing tags:" << endl;
	while ((ltag = lexer.scan()) != (void *) DONE) {
		cout << "\tLexTag: " << to_string(ltag) << endl;
		tags.push(ltag);
	}

	// Parser test
	Parser parser(&tags);

	cout << "Parser-------------------------->" << endl;
	parser.run();
	// parser.function();
	// parser.statement();
	parser.dump();

	// Free the elements of the queue
	while (!tags.empty()) {
		ltag = tags.pop();

		free_ltag(ltag);
	}
}
