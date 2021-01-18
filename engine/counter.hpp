#ifndef COUNTER_H_
#define COUNTER_H_

namespace zhetapi {

template <class T>
class Counter {
	T	__min;
	T	__max;
	T	__alpha;

	T	__count;
public:
	Counter(T, T, T);

	T operator()() const;
};

template <class T>
Counter <T> ::Counter(T mn, T mx, T alpha) : __min(mn), __max(mx),
		__alpha(alpha) {}

template <class T>
T Counter <T> ::operator()() const
{
	return (__count = min(max(__count + __alpha, __min), __max);
}

}

#endif
