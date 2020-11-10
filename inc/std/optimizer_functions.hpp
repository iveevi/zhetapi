#ifndef STD_OPTIMIZER_FUNCTIONS_H_
#define STD_OPTIMIZER_FUNCTIONS_H_

// C/C++ headers
#include <cstdlib>
#include <vector>

// Engine headers
#include <vector.hpp>

namespace zhetapi {
		
	namespace ml {

		// Squared error
		template <class T>
		Vector <T> __d_squared(const Vector <T> &comp, const Vector <T> &in)
		{
			return -T(2) * (comp - in);
		}

		template <class T>
		Vector <T> __squared(const Vector <T> &comp, const Vector <T> &in)
		{
			T sum = 0;

			for (size_t i = 0; i < comp.size(); i++)
				sum += (comp[i] - in[i]) * (comp[i] - in[i]);
			
			return {sum};
		}

		// Mean squared error
		template <class T>
		Vector <T> __d_mean_squared(const Vector <T> &comp, const Vector <T> &in)
		{
			return -T(2)/T(comp.size()) * (comp - in);
		}

		template <class T>
		Vector <T> __mean_squared(const Vector <T> &comp, const Vector <T> &in)
		{
			T sum = 0;

			for (size_t i = 0; i < comp.size(); i++)
				sum += (comp[i] - in[i]) * (comp[i] - in[i]);
					
			return std::vector <T> {sum/T(comp.size())};
		}

	}

}

#endif