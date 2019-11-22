// Standard C++ Libaries
#include <iostream>

// Custom headers
#include "types.h"

// Reused constants
#define ITEMS 5

using namespace std;
using namespace types;

int main()
{
    int oper;
    num_t first = num_t();
    num_t second = num_t();
    vector <num_t> operands;

    opn_t *select;

    for (int i = 0; i < ITEMS; i++) {
        cin >> first;
        cin >> second;
        
        cout << "Press 0 to add, 1 to subtract, 2 to multiply and 3 to divide: ";
        cin >> oper;

        switch(oper) {
        case 0:
            select = &(add_op <num_t>);
            break;
        case 1:
            select = &(sub_op <num_t>);
            break;
        case 2:
            select = &(mult_op <num_t>);
            break;
        case 3:
            select = &(div_op <num_t>);
            break;
        default:
            select = nullptr;
            break;
        }

        operands.push_back(first);
        operands.push_back(second);

        if (select == nullptr) {
            cout << "Valid option was not selected" << endl;
            continue;
        }
        
        try {
            cout << (*select)(operands) << endl;
        } catch (operation <num_t> ::argset_exception err) {
            cout << err.msg << endl;
        }

        operands.clear();
        cout.flush();
    }
}
