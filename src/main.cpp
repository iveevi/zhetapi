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
        cout << "Testing manually built tree:" << endl;
	token_tree <double> tr = token_tree <double> ();
        
        tr.set_cursor(ttwrapper <double> (module_t::opers[module_t::DIVOP]));
 
        tr.add_branch(ttwrapper <double> (operand <double> (45)));

	tr.add_branch(ttwrapper <double> (module_t::opers[module_t::ADDOP]));

        tr.move_down(1);
        tr.add_branch(ttwrapper <double> (operand <double> (213)));
        tr.add_branch(ttwrapper <double> (operand <double> (3123)));

	tr.print();
        IC(tr.value()->dptr);

        cout << endl << "Testing module parsing functions:";
        string str;

        cout << endl << "Enter a string to be parsed: ";
        cin >> str;

        cout << endl << "All tokens in string:" << endl;
        vector <token *> toks = module_t::get_tokens(str);

        for (token *t : toks)
                cout << "\t" << t->str() << endl;

        cout << endl << "Testing string parsed tree:" << endl;

        tr = token_tree <double> (toks);

        tr.print();
        IC(tr.value()->dptr);
}
