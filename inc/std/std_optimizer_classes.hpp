#ifndef STD_OPTIMIZER_CLASSES_H_
#define STD_OPTIMIZER_CLASSES_H_

// Engine headers
#include <optimizer.hpp>

// Engine std module headers
#include <std_optimizer_functions.hpp>

namespace ml {

	// Squared error
	template <class T>
	class __DSquaredError : public Optimizer <T> {
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) {
			std::cout << "D/DX SQUARED" << std::endl;
			return __d_squared <T> (comp, in);
		}
	};

	template <class T>
	class SquaredError : public Optimizer <T> {
		__DSquaredError <T> __deriv;
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) {
			std::cout << "SQUARED" << std::endl;
			return __squared <T> (comp, in);
		}

		Optimizer <T> &derivative()
		{
			return __deriv;
		}
	};

	// Mean squared error
	template <class T>
	class __DMeanSquaredError : public Optimizer <T> {
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) {
			std::cout << "D/DX MEAN SQUARED" << std::endl;
			return __d_mean_squared <T> (comp, in);
		}
	};

	template <class T>
	class MeanSquaredError : public Optimizer <T> {
		__DMeanSquaredError <T> __deriv;
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) {
			std::cout << "MEAN SQUARED" << std::endl;
			return __mean_squared <T> (comp, in);
		}

		Optimizer <T> &derivative()
		{
			return __deriv;
		}
	};

}

#endif
