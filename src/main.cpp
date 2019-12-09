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

// Debugging function
void dprint()
{
        while (true) {
                if (stage == -1)
                        return;
                cout << "stage = " << stage << endl;
                this_thread::sleep_for(chrono::microseconds(DELAY_P));
        }
}

extern int stage;

int main()
{
        // Debugging interface
        stage = 0;
        thread debug(dprint);

	token_tree <double> tr = token_tree <double> ();
        stage = 1;

	tr.add_branch(new ttwrapper <double> (operand <double> {45}));
	tr.add_branch(new ttwrapper <double> (operand <double> {67}));
	stage = 2;

	tr.move_down(0);
        stage = 3;

	tr.add_branch(new ttwrapper <double> (operand <double> {90}));
        stage = 4;

	tr.move_up();
	tr.move_right();
        stage = 5;

	tr.add_branch(new ttwrapper <double> (operand <double> {90}));
	tr.add_branch(new ttwrapper <double> (operand <double> {120}));
        stage = -1;

        // Wait for thread to exit
        this_thread::sleep_for(chrono::seconds(DELAY));
        cout << "Printing..." << endl;
	tr.print();
}
