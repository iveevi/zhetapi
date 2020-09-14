#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// C/C++ headers
#include <cmath>
#include <memory>

// Engine headers
#include <vector.hpp>

namespace ml {

	template <class T>
	class Optimizer {
	public:
		Optimizer();

		virtual Vector <T> operator()(const Vector <T> &, const Vector <T> &) const;

		virtual std::shared_ptr <Optimizer> derivative();
	};

	template <class T>
	Optimizer <T> ::Optimizer() {}

	template <class T>
	Vector <T> Optimizer <T> ::operator()(const Vector <T> &comp, const Vector <T> &in) const
	{
		std::cout << "HERE" << std::endl;
		return {(comp - in).norm()};
	}

	template <class T>
	std::shared_ptr <Optimizer <T>> Optimizer <T> ::derivative()
	{
		return new Optimizer();
	}

	// test
	template <class T>
	class DTest : public Optimizer <T> {
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
			std::cout << "DERIV!!" << std::endl;

			return in;
		}
	};

	template <class T>
	class Test : public Optimizer <T> {
	public:
		Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
			std::cout << "TEST!!" << std::endl;

			return comp;
		}

		std::shared_ptr <Optimizer <T>> derivative()
		{
			return new DTest <T> ();
		}
	};

}

#endif
