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
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) {
			std::cout << "SQUARED" << std::endl;
			return __squared <T> (comp, in);
		}

		__DSquaredError <T> *derivative() const override
		{
			return new __DSquaredError <T> ();
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
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) {
			std::cout << "MEAN SQUARED" << std::endl;
			return __mean_squared <T> (comp, in);
		}

		__DMeanSquaredError <T> *derivative() const
		{
			return new __DMeanSquaredError <T> ();
		}
	};

}

#endif