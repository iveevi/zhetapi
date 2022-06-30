#ifndef ZHETAPI_AUTOGRAD_TRAIN_H_
#define ZHETAPI_AUTOGRAD_TRAIN_H_

// Extra headers
#include <indicators/progress_bar.hpp>

// Library headers
#include "function.hpp"
#include "optimizer.hpp"
#include "autograd.hpp"

namespace zhetapi {

namespace autograd {

namespace ml {

// Useful aliases
using Data = std::vector <_function::Input>;

// Progress reporters
struct _reporter {
	struct Info {
		size_t epoch;
		size_t batch;
		size_t total_epochs;
	};

	virtual void report(const Info &) const {}
};

// Progress bar
class ProgressBar : public _reporter {
	mutable indicators::ProgressBar bar;
public:
	ProgressBar() : bar {
		indicators::option::BarWidth {50},
		indicators::option::Start {" ["},
		indicators::option::Fill {"█"},
		indicators::option::Lead {"█"},
		indicators::option::Remainder {"-"},
		indicators::option::End {"]"},
		indicators::option::PrefixText {"Training model"},
		indicators::option::ShowElapsedTime {true},
		indicators::option::ShowRemainingTime {true}
	} {}

	void report(const Info &info) const override {
		float progress = 100.0f * (float) info.epoch / info.total_epochs;
		bar.set_progress(progress);
	}
};

// TODO: source file
// TODO: some function/interface to report progress
// TODO: default optimizer and loss function
inline void fit(_function &f, const Data &X, const std::vector <Constant> &Y,
		_optimizer &optimizer, _function &loss,
		size_t iterations = 1000,
		const _reporter &reporter = ProgressBar())
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

		reporter.report({i, 0, iterations});
	}
}

}

}

}

#endif
