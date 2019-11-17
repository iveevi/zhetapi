// Standard C++ Libaries
#include <iostream>

// Custom headers
#include "operand.h"
#include "operation.h"

// Reused constants
#define ITEMS 10

using namespace std;
using namespace operands;
using namespace operations;

int main()
{
    operand <def_t> oper = operand <def_t> ();
    
    for (int i; i < ITEMS; i++) {
        cin >> oper;
        cout << "Operand: " << oper << endl;
    }
}