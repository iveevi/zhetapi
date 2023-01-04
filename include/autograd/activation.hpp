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

		Gradient gradient(const Input &ins, const Input &igrads) override {
			Constant out = Constant(igrads[0].shape(),
				[&](size_t i) {
					float x = ins[0].get(i);
					return (x > 0 ? 1 : 0) * igrads[0].get(i);
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "RELU";
		}

		Ptr copy() const override {
			return Ptr(new_ftn_ <kernel> ());
		}
	};

	_relu() : ISeq(new_ftn_ <kernel> (), 1) {}
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

		Gradient gradient(const Input &ins, const Input &igrads) override {
			Constant out = Constant(igrads[0].shape(),
				[&](size_t i) {
					float x = ins[0].get(i);
					return (x > 0 ? 1 : _alpha) * igrads[0].get(i);
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "LEAKY RELU";
		}

		Ptr copy() const override {
			return Ptr(new_ftn_ <kernel> (_alpha));
		}
	};

	_leaky_relu(float alpha) : ISeq(new_ftn_ <kernel> (alpha), 1) {}
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

		Gradient gradient(const Input &ins, const Input &igrads) override {
			Constant out = Constant(igrads[0].shape(),
				[&](size_t i) {
					float x = ins[0].get(i);
					float y = 1 / (1 + std::exp(-x));
					return y * (1 - y) * igrads[0].get(i);
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "SIGMOID";
		}

		Ptr copy() const override {
			return Ptr(new_ftn_ <kernel> ());
		}
	};

	_sigmoid() : ISeq(new_ftn_ <kernel> (), 1) {}
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

		Gradient gradient(const Input &ins, const Input &igrads) override {
			Constant out = Constant(igrads[0].shape(),
				[&](size_t i) {
					float x = ins[0].get(i);
					float y = std::tanh(x);
					return (1 - y * y) * igrads[0].get(i);
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "TANH";
		}

		Ptr copy() const override {
			return Ptr(new_ftn_ <kernel> ());
		}
	};

	_tanh() : ISeq(new_ftn_ <kernel> (), 1) {}
};

extern Function tanh;

// Softmax activation function
class _softmax : public ISeq {
public:
	struct kernel : public _function {
		kernel() : _function(1) {}

		Constant compute(const Input &ins) override {
			auto o = ins[0].copy();

			float omax = max(o);
			o -= omax;

			float osum = sum(o, expf);
			return o.transform(
				[osum](float x) {
					return std::exp(x)/osum;
				}
			);
		}

		Gradient gradient(const Input &ins, const Input &igrads) override {
			auto o = ins[0].copy();

			auto omax = max(o);
			o -= omax;

			auto osum = sum(o, expf);

			Constant out({o.size()},
				[&](size_t i) {
					float x = o.get(i);
					float e = std::exp(x);
					float t = e * (osum - e) / (osum * osum);
					return t * igrads[0].get(i);
				}
			);

			return Gradient {
				.igrads = {out}
			};
		}

		std::string summary() const override {
			return "SOFTMAX";
		}

		Ptr copy() const override {
			return Ptr(new_ftn_ <kernel> ());
		}
	};

	_softmax() : ISeq(new_ftn_ <kernel> (), 1) {}
};

extern Function softmax;

}

}

}
