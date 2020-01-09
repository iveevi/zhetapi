// C++ standard libraries
#include <chrono>
#include <iostream>
#include <string>
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
        token_tree <double> tr;
        string input;

        while (true) {
                cout << "Enter Input: ";
                getline(cin, input);

                if (input.empty())
                        break;
                tr = token_tree <double> (input);

                tr.print();
                dp_ptr(tr.value()->dptr);
        }
}
