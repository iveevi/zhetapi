#include "port.hpp"

// Testing rig
vector <pair <string, bool(*)()>> rig {
	{"vector construction and memory safety", &vector_construction_and_memory},
	{"matrix construction and memory safety", &matrix_construction_and_memory},
	{"tensor construction and memory safety", &tensor_construction_and_memory},
	{"function general compilation", &function_compilation_testing}
};

// Segfault handler
void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    printf("\nCaught segfault at address %p\n", si->si_addr);
    exit(-1);
}

// Timers
chrono::high_resolution_clock clk;

chrono::high_resolution_clock::time_point epoch;
chrono::high_resolution_clock::time_point tmp;

// Main program
int main()
{
	// Setup segfault handler
	struct sigaction sa;

	memset(&sa, 0, sizeof(struct sigaction));

	sigemptyset(&sa.sa_mask);

	sa.sa_sigaction = segfault_sigaction;
	sa.sa_flags   = SA_SIGINFO;

	sigaction(SIGSEGV, &sa, NULL);

	// Setup times
	epoch = clk.now();

	// Run tests in the test rig
	bool first = true;

	int count = 0;
	for (auto pr : rig) {
		if (first)
			first = false;
		else
			cout << endl;

		cout << "==============================" << endl;
		cout << mark << "Running \"" << pr.first << "\" test:" << endl;

		cout << endl << "--------------------" << endl;
		bool tmp = pr.second();
		cout << "--------------------" << endl;

		if (tmp) {
			cout << endl << "\"" << pr.first << "\" test PASSED." << endl;
			count++;
		} else {
			cout << endl << "\"" << pr.first << "\" test FAILED." << endl;
		}

		cout << "==============================" << endl;
	}

	cout << endl << mark << "Summary: passed " << count << "/" << rig.size() << " tests." << endl;
}
