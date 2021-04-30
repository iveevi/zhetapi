#ifndef COUNTER_H_
#define COUNTER_H_

namespace zhetapi {

template <class T>
class Counter {
	T	_min;
	T	_max;
	T	_alpha;

	T	_count;
public:
	Counter(T, T, T);

	T operator()() const;
};

template <class T>
Counter <T> ::Counter(T mn, T mx, T alpha) : _min(mn), _max(mx),
		_alpha(alpha) {}

template <class T>
T Counter <T> ::operator()() const
{
	return (_count = min(max(_count + _alpha, _min), _max));
}

}

#endif
