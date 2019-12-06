// C++ standard libraries
#include <iostream>

// Custom made libraries
#include "trees.h"
#include "tokens.h"

// Including namespaces
using namespace std;
using namespace tokens;
using namespace trees;

int main()
{
	token_tree <double> tr = token_tree <double> ();

	tr.add_branch(new operand <int> (45));
	tr.add_branch(new operand <int> (67));
	
	tr.move_down(0);

	tr.add_branch(new operand <int> (90));

	tr.move_up();
	tr.move_right();

	tr.add_branch(new operand <int> (90));
	tr.add_branch(new operand <int> (120));

	tr.print();
}
