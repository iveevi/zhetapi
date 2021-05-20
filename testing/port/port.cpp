#include "port.hpp"

#define THREADS	8

typedef pair <string, bool (*)(ostringstream &)> singlet;

// Testing rig
vector <singlet> rig {
	RIG(gamma_and_factorial),
	RIG(vector_construction_and_memory),
	RIG(matrix_construction_and_memory),
	RIG(tensor_construction_and_memory),
	RIG(integration),
	RIG(function_computation),
	RIG(vector_operations),
	RIG(interval_construction),
	RIG(interval_sampling),
	RIG(diag_matrix),
	RIG(qr_decomp),
	RIG(lq_decomp),
	RIG(qr_alg),
	RIG(matrix_props),
	RIG(fourier_series),
	RIG(polynomial_construction),
	RIG(polynomial_comparison),
	RIG(polynomial_arithmetic),
	RIG(act_linear),
	RIG(act_relu),
	RIG(act_leaky_relu),
	RIG(act_sigmoid),
	RIG(module_construction),
	RIG(parsing_global_assignment),
	RIG(parsing_global_branching)
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

		bool tmp = true;
		
		try {
			oss << string(100, '-') << endl;
			tmp = s.second(oss);	
			oss << string(100, '-') << endl;
		} catch (...) {
			cout << bred << "CAUGHT UNKNOWN EXCEPTION, TERMINATING."
				<< reset << endl;
			
			throw;
		}

		if (tmp) {
			oss << endl << bgreen << "\"" << s.first
				<< "\" test PASSED." << reset << endl;
		} else {
			// Add to list of failed tasks
			fl_mtx.lock();

			failed.push_back(s);

			fl_mtx.unlock();

			oss << endl << bred << "\"" << s.first
				<< "\" test FAILED." << reset << endl;
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

			if (task < (int) size) {
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

	return (failed.size() == 0) ? 0 : 1;
}
