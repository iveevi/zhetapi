#include "autograd.hpp"
#include "function.hpp"
#include "iseq.hpp"

namespace zhetapi {

namespace autograd {

namespace ml {

// RELU activation function
class _relu : public ISeq {
public:
	struct kernel : public _function {
		kernel() : _function(1) {}

		Constant compute(const Input &ins) override {
			return ins[0].copy().transform(
				[](long double x) {
					return x > 0 ? x : 0;
				}
			);
		}

		Gradient gradient(const Input &igrads) const override {
			Constant out = igrads[0].copy().transform(
				[](long double x) {
					return (x > 0 ? 1 : 0) * x;
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "RELU";
		}

		_function *copy() const override {
			return new kernel();
		}
	};

	_relu() : ISeq(new kernel(), 1) {}
};

extern Function relu;

// Leaky RELU activation function
class _leaky_relu : public ISeq {
public:
	struct kernel : public _function {
		long double _alpha;

		kernel(long double alpha) : _function(1), _alpha(alpha) {}

		Constant compute(const Input &ins) override {
			return ins[0].copy().transform(
				[this](long double x) {
					return x > 0 ? x : _alpha * x;
				}
			);
		}

		Gradient gradient(const Input &igrads) const override {
			Constant out = igrads[0].copy().transform(
				[this](long double x) {
					return (x > 0 ? 1 : _alpha) * x;
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "Leaky RELU";
		}

		_function *copy() const override {
			return new kernel(_alpha);
		}
	};

	_leaky_relu(long double alpha) : ISeq(new kernel(alpha), 1) {}
};

inline Function leaky_relu(long double alpha)
{
	return new_ <_leaky_relu> (alpha);
}

}

}

}
