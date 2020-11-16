#ifndef STD_OPTIMIZER_CLASSES_H_
#define STD_OPTIMIZER_CLASSES_H_

// Engine headers
#include "../optimizer.hpp"

// Engine std module headers
#include "optimizer_functions.hpp"

namespace zhetapi {
		
	namespace ml {
		
		/*
		* All optimizer classes have inlined member functions for the same
		* reason that the activation classes are inlined. Obscure naming is
		* also done for the same reason.
		*/

		// Squared error
		template <class T>
		class __DSquaredError : public Optimizer <T> {
		public:
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __d_squared <T> (comp, in);
			}
		};

		template <class T>
		class SquaredError : public Optimizer <T> {
		public:
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __squared <T> (comp, in);
			}

			Optimizer <T> *derivative() const
			{
				return new __DSquaredError <T> ();
			}
		};

		// Mean squared error
		template <class T>
		class __DMeanSquaredError : public Optimizer <T> {
		public:
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __d_mean_squared <T> (comp, in);
			}
		};

		template <class T>
		class MeanSquaredError : public Optimizer <T> {
		public:
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __mean_squared <T> (comp, in);
			}

			Optimizer <T> *derivative() const
			{
				return new __DMeanSquaredError <T> ();
			}
		};

	}

}

#endif
