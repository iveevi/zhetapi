#ifndef ZHETAPI_AUTOGRAD_TRAIN_H_
#define ZHETAPI_AUTOGRAD_TRAIN_H_

// Standard headers
#include <vector>

// Extra headers
#include <indicators/progress_bar.hpp>

// Library headers
#include "function.hpp"
#include "optimizer.hpp"
#include "autograd.hpp"

#include <iostream>

namespace zhetapi {

namespace autograd {

namespace ml {

// Useful aliases
using Data = std::vector <_function::Input>;
using Validator = std::function <bool (const Constant &, const Constant &)>;

// Check accuracy of model wrt data set, returns proportion
// TODO: source file
inline float accuracy(_function &model,
		const Data &X, const std::vector <Constant> &Y,
		const Validator &validator)
{
	int count = 0;
	std::vector <int> indices;
	for (int i = 0; i < X.size(); i++) {
		Constant output = model.compute(X[i]);
		if (validator(output, Y[i])) {
			indices.push_back(i);
			count++;
		}
	}

	return float(count) / X.size();
}

// Progress reporters
struct _reporter {
	struct Info {
		size_t		epoch;
		size_t		batch;
		size_t		total_epochs;
		float		avg_error;
		_function	&model;
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

// Validate (requires data set)
class Validate : public _reporter {
	const Data &X;
	const std::vector <Constant> &Y;
	Validator validator;
public:
	Validate(const Data &X_, const std::vector <Constant> &Y_, const Validator &validator_)
		: X {X_}, Y {Y_}, validator {validator_} {}

	void report(const Info &info) const override {
		float accuracy = zhetapi::autograd::ml::accuracy(info.model, X, Y, validator);
		std::cout << "Accuracy: " << accuracy
			<< ", Average error = " << info.avg_error << std::endl;

		// TODO: make this optional
		detail::MemoryTracker::report();
	}
};

// TODO: source file
// TODO: some function/interface to report progress
// TODO: default optimizer and loss function

// Information relevant to training
struct TrainingSuite {
	_function &loss;
	_function &dloss;
	size_t iterations;
	size_t batch_size;
	std::shared_ptr <_reporter> reporter = std::make_shared <ProgressBar> ();
};

inline void fit(_function &f, const Data &X, const std::vector <Constant> &Y,
		_optimizer &optimizer, const TrainingSuite &suite)
{
	// TODO: assert that X.size() == Y.size()
	for (size_t i = 0; i < suite.iterations; i++) {
		// TODO: implement verbose
		GradientQueue gq;
		int elements = 0;

		float serror = 0;
		for (size_t j = 0; j < X.size(); j++) {
			Constant y = f.compute(X[j]).flat();
			Constant igrad = suite.dloss.compute({y, Y[j]});
			_function::Gradient grads = f.gradient(X[j], {igrad});
			serror += suite.loss.compute({y, Y[j]}).length();

			elements++;
			if (gq.empty())
				gq = grads.grads;
			else
				gq += grads.grads;

			if (elements >= suite.batch_size) {
				gq /= float(elements);
				optimizer.optimize(gq);
				f.update_parameters(gq);

				gq.clear();
				elements = 0;
			}
		}

		suite.reporter->report({i, 0, suite.iterations, serror/X.size(), f});
	}
}

}

}

}

#endif
