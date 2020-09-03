#ifndef STD_OPTIMIZER_CLASSES_H_
#define STD_OPTIMIZER_CLASSES_H_

// Engine headers
#include <optimizer.hpp>

// Engine std module headers
#include <std_optimizer_functions.hpp>

namespace ml {

	template <class T>
	class SquaredError {
	public:
		T operator()(const Vector <T> &comp, const Vector <T> &in) {
			return __squared <T> (comp, in);
		}
	};

	template <class T>
	class MeanSquaredError {
	public:
		T operator()(const Vector <T> &comp, const Vector <T> &in) {
			return __mean_squared <T> (comp, in);
		}
	};

}

#endif