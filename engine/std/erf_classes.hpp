#ifndef STD_Erf_CLASSES_H_
#define STD_Erf_CLASSES_H_

// Engine headers
#include <erf.hpp>

// Engine std module headers
#include <std/erf_functions.hpp>

namespace zhetapi {
		
	namespace ml {
		
		/*
		* All Erf classes have inlined member functions for the same
		* reason that the activation classes are inlined. Obscure naming is
		* also done for the same reason.
		*/

		// Squared error
		template <class T>
		class __DSquaredError : public Erf <T> {
		public:

#ifndef ZHP_CUDA

			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __d_squared <T> (comp, in);
			}

#else

			__host__ __device__
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return -T(2) * (comp - in);
			}

#endif

		};

		template <class T>
		class SquaredError : public Erf <T> {
		public:

#ifndef ZHP_CUDA

			SquaredError() {
				this->kind = Erf <T> ::OPT_SE; 
			}

			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __squared <T> (comp, in);
			}

			Erf <T> *derivative() const
			{
				return new __DSquaredError <T> ();
			}

#else

			__host__ __device__
			SquaredError() {
				this->kind = Erf <T> ::OPT_SE;
			}

			__host__ __device__
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				T sum = 0;

				for (size_t i = 0; i < comp.size(); i++)
					sum += (comp[i] - in[i]) * (comp[i] - in[i]);
				
				return Vector <T> (1, sum);
			}

			__host__ __device__
			Erf <T> *derivative() const
			{
				return new __DSquaredError <T> ();
			}

#endif

		};

		// Mean squared error
		template <class T>
		class __DMeanSquaredError : public Erf <T> {
		public:

#ifndef ZHP_CUDA

			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __d_mean_squared <T> (comp, in);
			}

#else

			__host__ __device__
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return -T(2)/T(comp.size()) * (comp - in);
			}

#endif

		};

		template <class T>
		class MeanSquaredError : public Erf <T> {
		public:

#ifndef ZHP_CUDA

			MeanSquaredError()
			{
				this->kind = Erf <T> ::OPT_MSE;
			}
			
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				return __mean_squared <T> (comp, in);
			}

			Erf <T> *derivative() const
			{
				return new __DMeanSquaredError <T> ();
			}

#else

			__host__ __device__
			MeanSquaredError() {
				this->kind = Erf <T> ::OPT_MSE;
			}

			__host__ __device__
			Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
				T sum = 0;

				for (size_t i = 0; i < comp.size(); i++)
					sum += (comp[i] - in[i]) * (comp[i] - in[i]);
				
				return Vector <T> (1, sum / T(comp.size()));
			}

			__host__ __device__
			Erf <T> *derivative() const {
				return new __DMeanSquaredError <T> ();
			}

#endif

		};

#ifdef ZHP_CUDA

		// Copy base activations
		template <class T>
		__host__ __device__
		Erf <T> *copy(Erf <T> *opt)
		{
			switch (opt->kind) {
			case Erf <T> ::OPT_Default:
				return new Erf <T> ();
			case Erf <T> ::OPT_SE:
				return new SquaredError <T> ();
			case Erf <T> ::OPT_MSE:
				return new MeanSquaredError <T> ();
			}

			return nullptr;
		}

#endif

	}

}

#endif
