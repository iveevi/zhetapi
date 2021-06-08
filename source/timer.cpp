#include <timer.hpp>

namespace zhetapi {

Timer::Timer() {}

void Timer::start()
{
	_start = _clk.now();
}

void Timer::stop()
{
	_end = _clk.now();
}

Timer::time Timer::now()
{
	return _clk.now(); 
}

long double Timer::dt()
{
	return (std::chrono::duration_cast <std::chrono::microseconds>
		(_end - _start)).count();
}

long double Timer::split()
{
	stop();

	return dt();
}

}
