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
				[](float x) {
					return x > 0 ? x : 0;
				}
			);
		}

		Gradient gradient(const Input &igrads) const override {
			Constant out = igrads[0].copy().transform(
				[](float x) {
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
		float _alpha;

		kernel(float alpha) : _function(1), _alpha(alpha) {}

		Constant compute(const Input &ins) override {
			return ins[0].copy().transform(
				[this](float x) {
					return x > 0 ? x : _alpha * x;
				}
			);
		}

		Gradient gradient(const Input &igrads) const override {
			Constant out = igrads[0].copy().transform(
				[this](float x) {
					return (x > 0 ? 1 : _alpha) * x;
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "LEAKY RELU";
		}

		_function *copy() const override {
			return new kernel(_alpha);
		}
	};

	_leaky_relu(float alpha) : ISeq(new kernel(alpha), 1) {}
};

inline Function leaky_relu(float alpha)
{
	return new_ <_leaky_relu> (alpha);
}

// Sigmoid activation function
class _sigmoid : public ISeq {
public:
	struct kernel : public _function {
		kernel() : _function(1) {}

		Constant compute(const Input &ins) override {
			return ins[0].copy().transform(
				[](float x) {
					return 1 / (1 + std::exp(-x));
				}
			);
		}

		Gradient gradient(const Input &igrads) const override {
			Constant out = igrads[0].copy().transform(
				[](float x) {
					float y = 1 / (1 + std::exp(-x));
					return y * (1 - y) * x;
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "SIGMOID";
		}

		_function *copy() const override {
			return new kernel();
		}
	};

	_sigmoid() : ISeq(new kernel(), 1) {}
};

extern Function sigmoid;

// Tanh activation function
class _tanh : public ISeq {
public:
	struct kernel : public _function {
		kernel() : _function(1) {}

		Constant compute(const Input &ins) override {
			return ins[0].copy().transform(
				[](float x) {
					return std::tanh(x);
				}
			);
		}

		Gradient gradient(const Input &igrads) const override {
			Constant out = igrads[0].copy().transform(
				[](float x) {
					float y = std::tanh(x);
					return (1 - y * y) * x;
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "TANH";
		}

		_function *copy() const override {
			return new kernel();
		}
	};

	_tanh() : ISeq(new kernel(), 1) {}
};

extern Function tanh;

}

}

}
