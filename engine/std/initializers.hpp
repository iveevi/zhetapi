#ifndef STD_INITIALIZERS_H_
#define STD_INITIALIZERS_H_

#ifdef __AVR	// Does not support AVR

#include "avr/random.hpp"

#else

// C++ headers
#include <random>

#endif		// Does not support AVR

namespace zhetapi {

namespace ml {

#ifdef __AVR

template <class T>
T RandomInitializer() {
	static avr::RandomEngine reng(16183);
	
	return reng.ldouble();
}

#else

template <class T>
struct RandomInitializer {
	// Use interval later
	T operator()() {
		return T (0.5 - rand()/((double) RAND_MAX));
	}
};

#endif

#ifndef __AVR	// Does not support AVR

template <class T>
struct LeCun {
	std::default_random_engine		_gen;
	std::normal_distribution <double>	_dbt;	
public:
	explicit LeCun(size_t fan_in) : _gen(),
			_dbt(0, sqrt(T(1) / fan_in)) {}
	
	T operator()() {
		return _dbt(_gen);
	}
};

template <class T>
struct He {
	std::default_random_engine		_gen;
	std::normal_distribution <double>	_dbt;	
public:
	explicit He(size_t fan_in) : _gen(),
			_dbt(0, sqrt(T(2) / fan_in)) {}
	
	T operator()() {
		return _dbt(_gen);
	}
}; 

template <class T>
struct Xavier {
	std::default_random_engine		_gen;
	std::normal_distribution <double>	_dbt;	
public:
	explicit Xavier(size_t fan_avg) : _gen(),
			_dbt(0, sqrt(T(1) / fan_avg)) {}
	
	T operator()() {
		return _dbt(_gen);
	}
};

#endif

}

}

#endif
