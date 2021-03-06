#include "port.hpp"

#define THREADS	8

typedef pair <string, bool (*)(ostringstream &)> singlet;

// Testing rig
vector <singlet> rig {
	RIG(gamma_and_factorial),
	RIG(vector_construction_and_memory),
	RIG(function_compilation_testing),
	RIG(matrix_construction_and_memory),
	RIG(tensor_construction_and_memory),
	RIG(integration),
	RIG(function_computation),
	RIG(vector_operations),
	RIG(interval_construction),
	RIG(interval_sampling),
	RIG(diag_matrix),
	RIG(qr_decomp),
	RIG(qr_alg)
};

vector <singlet> failed;

// Segfault handler
void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
	printf("\nCaught segfault at address %p\n", si->si_addr);
	exit(-1);
}

// Timers
tclk clk;

// Main program
int main()
{
	// Setup segfault handler
	struct sigaction sa;

	memset(&sa, 0, sizeof(struct sigaction));

	sigemptyset(&sa.sa_mask);

	sa.sa_sigaction = segfault_sigaction;
	sa.sa_flags = SA_SIGINFO;

	sigaction(SIGSEGV, &sa, NULL);

	// Setup times
	tpoint epoch = clk.now();

	bench mark(epoch);

	mutex io_mtx;	// I/O mutex
	mutex tk_mtx;	// Task acquisition mutex
	mutex fl_mtx;	// Task failure mutex

	int count = 0;
	int task = 0;

	size_t size = rig.size();
	auto singleter = [&](singlet s, size_t t) {
		ostringstream oss;
		
		oss << string(100, '=') << endl;
		oss << mark << "Running \"" << s.first
			<< "\" test [" << t << "/"
			<< size << "]:\n" << endl;

		oss << string(100, '-') << endl;
		bool tmp = s.second(oss);	
		oss << string(100, '-') << endl;

		if (tmp) {
			oss << endl << "\"" << s.first
				<< "\" test PASSED." << endl;
		} else {
			// Add to list of failed tasks
			fl_mtx.lock();

			failed.push_back(s);

			fl_mtx.unlock();

			oss << endl << "\"" << s.first
				<< "\" test FAILED." << endl;
		}
		
		oss << string(100, '=') << endl;
		
		io_mtx.lock();

		cout << oss.str() << endl;
		count += (tmp) ? 1 : 0;

		io_mtx.unlock();
	};

	auto tasker = [&]() {
		while (true) {
			int t = -1;

			tk_mtx.lock();

			if (task < size) {
				t = task;

				task++;
			}

			tk_mtx.unlock();

			if (t < 0)
				break;

			singleter(rig[t], t + 1);
		}
	};

	thread *army = new thread[THREADS];
	for (size_t i = 0; i < THREADS; i++)
		army[i] = thread(tasker);

	for (size_t i = 0; i < THREADS; i++)
		army[i].join();

	cout << endl << mark << "Summary: passed "
		<< count << "/" << rig.size()
		<< " tests." << endl;

	if (failed.size()) {
		cout << endl << string(100, '=') << endl;

		cout << "Failed tests [" << failed.size() 
			<< "/" << rig.size() << "]:" << endl;

		for (auto task : failed) {
			cout << "\t" << task.first << endl;
		}

		cout << string(100, '=') << endl;
	}
}
