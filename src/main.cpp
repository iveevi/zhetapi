// C++ standard libraries
#include <chrono>
#include <iostream>
#include <thread>

// Custom made libraries
#include "trees.h"
#include "tokens.h"

#include "debug.h"

// Including namespaces
using namespace std;
using namespace tokens;
using namespace trees;

// Delay time
#define DELAY 1
#define DELAY_P 100

extern int stage;

int main()
{
	token_tree <double> tr = token_tree <double> ();
        
        tr.set_cursor(new ttwrapper <double> (module <operand <double>>
        ::opers[module <operand <double>> ::ADDOP]));
 
        tr.add_branch(new ttwrapper <double> (operand <double> (45)));
	tr.add_branch(new ttwrapper <double> (operand <double> (345)));

	tr.print();
        IC(tr.value()->dptr);
}
