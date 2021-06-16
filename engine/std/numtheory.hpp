#ifndef NUMBER_THEORY_H_
#define NUMBER_THEORY_H_

// C/C++ headers
#include <cmath>
#include <vector>

namespace zhetapi {

namespace numtheory {

// TODO: change to only integral types
template <class T = long long>
std::vector <T> sieve(T lim)
{
	std::vector <T> primes = {2};
	
	if (lim < 2)
		return {};

	for (T i = 3; i < lim; i++) {
		bool prime = true;
		
		for (size_t j = 0; primes[j] <= sqrt(i); j++) {
			if (i % primes[j] == 0) {
				prime = false;

				break;
			}
		}

		if (prime)
			primes.push_back(i);
	}

	return primes;
}

}

}

#endif
