#include "port.hpp"

#define THREADS	8

typedef pair <string, bool (*)(ostringstream &)> singlet;

// Testing rig
vector <pair <string, bool(*)(ostringstream &)>> rig {
	{"gamma and factorial functions", &gamma_and_factorial},
	{"vector construction and memory safety", &vector_construction_and_memory},
	{"function general compilation", &function_compilation_testing},
	{"matrix construction and memory safety", &matrix_construction_and_memory},
	{"tensor construction and memory safety", &tensor_construction_and_memory},
	{"integration techniques", &integration},
	{"function computation", &function_computation},
	{"vector operations", &vector_operations},
	{"interval construction", &interval_construction},
	{"interval sampling", &interval_sampling}
};

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

	int count = 0;
	int task = 0;

	size_t size = rig.size();
	auto singleter = [&](singlet s, size_t t) {
		ostringstream oss;
		
		oss << string(100, '=') << endl;
		oss << mark << "Running \"" << s.first << "\" test: [" << t << "/" << size << "]\n" << endl;

		oss << string(100, '-') << endl;
		bool tmp = s.second(oss);	
		oss << string(100, '-') << endl;

		if (tmp)
			oss << endl << "\"" << s.first << "\" test PASSED." << endl;
		else
			oss << endl << "\"" << s.first << "\" test FAILED." << endl;
		
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

	cout << endl << mark << "Summary: passed " << count << "/" << rig.size() << " tests." << endl;
}
