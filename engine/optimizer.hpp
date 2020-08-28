#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// C/C++ headers
#include <cmath>

// Engine headers
#include <matrix.hpp>

namespace ml {

	template <class T>
	class Optimizer {
	public:
		using ftr = T (*)(const Vector <T> &);
	private:
		ftr
	};

}

#endif