#ifndef ZHETAPI_AUTOGRAD_GQ_H_
#define ZHETAPI_AUTOGRAD_GQ_H_

// Standard headers
#include <deque>

// Library headers
#include "../tensor.hpp"

namespace zhetapi {

namespace autograd {

// Constants are just tensors
using Constant = Tensor <float>;

// GradientQueue class is a deque with extra operations
class GradientQueue : public std::deque <Constant> {
public:
	// Constructors
	GradientQueue() = default;

	// Initializer list
	GradientQueue(std::initializer_list <Constant> l)
		: std::deque <Constant> (l) {}

	// Arithematic operations
	GradientQueue &operator+=(const GradientQueue &rhs) {
		assert(size() == rhs.size());
		for (size_t i = 0; i < size(); i++)
			at(i) += rhs[i];
		return *this;
	}

	GradientQueue &operator-=(const GradientQueue &rhs) {
		assert(size() == rhs.size());
		for (size_t i = 0; i < size(); i++)
			at(i) -= rhs[i];
		return *this;
	}

	GradientQueue &operator*=(const GradientQueue &rhs) {
		assert(size() == rhs.size());
		for (size_t i = 0; i < size(); i++)
			at(i) *= rhs[i];
		return *this;
	}

	GradientQueue &operator/=(const GradientQueue &rhs) {
		assert(size() == rhs.size());
		for (size_t i = 0; i < size(); i++)
			at(i) /= rhs[i];
		return *this;
	}

	// Single constant operations
	GradientQueue &operator+=(const Constant &rhs) {
		for (auto &x : *this)
			x += rhs;
		return *this;
	}

	GradientQueue &operator-=(const Constant &rhs) {
		for (auto &x : *this)
			x -= rhs;
		return *this;
	}

	GradientQueue &operator*=(const Constant &rhs) {
		for (auto &x : *this)
			x *= rhs;
		return *this;
	}

	GradientQueue &operator/=(const Constant &rhs) {
		for (auto &x : *this)
			x /= rhs;
		return *this;
	}
};

// More operators
// TODO: source file
inline GradientQueue operator*(const GradientQueue &lhs, const float &rhs)
{
	GradientQueue gq = lhs;
	gq *= rhs;
	return gq;
}

inline GradientQueue operator*(const float &lhs, const GradientQueue &rhs)
{
	GradientQueue gq = rhs;
	gq *= lhs;
	return gq;
}

inline GradientQueue operator/(const GradientQueue &lhs, const float &rhs)
{
	GradientQueue gq = lhs;
	gq /= rhs;
	return gq;
}

inline GradientQueue operator/(const float &lhs, const GradientQueue &rhs)
{
	GradientQueue gq = rhs;
	gq /= lhs;
	return gq;
}

}

}

#endif
