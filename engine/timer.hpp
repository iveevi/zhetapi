#ifndef TIMER_H_
#define TIMER_H_

// C/C++ headers
#include <chrono>

namespace zhetapi {

class Timer {
public:
	using clk = std::chrono::high_resolution_clock;
	using time = clk::time_point;
private:
	clk	_clk;
	time	_start;
	time	_end;
public:
	Timer();

	void start();
	void stop();

	time now();

	long double dt();
	long double split();
};

}

#endif
