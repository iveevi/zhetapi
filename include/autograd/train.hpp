#ifndef ZHETAPI_AUTOGRAD_TRAIN_H_
#define ZHETAPI_AUTOGRAD_TRAIN_H_

#include "function.hpp"
#include "optimizer.hpp"
#include "autograd.hpp"

namespace zhetapi {

namespace autograd {

namespace ml {

// Useful aliases
using Data = std::vector <_function::Input>;

// TODO: source file
// TODO: some function/interface to report progress
// TODO: default optimizer and loss function
inline void fit(_function &f, const Data &X, const std::vector <Constant> &Y,
		_optimizer &optimizer, _function &loss,
		size_t iterations = 1000, bool verbose = false)
{
	// TODO: assert that X.size() == Y.size()
	float inv = 1.0/X.size();
	for (size_t i = 0; i < iterations; i++) {
		// TODO: implement verbose
		GradientQueue gq;
		for (size_t j = 0; j < X.size(); j++) {
			Constant y = f.compute(X[j]).flat();
			Constant igrad = loss.compute({y, Y[j]});
			_function::Gradient grads = f.gradient({igrad});

			if (gq.empty())
				gq = grads.grads;
			else
				gq += grads.grads * inv;
		}

		optimizer.optimize(gq);
		f.update_parameters(gq);
	}
}

}

}

}

#endif
