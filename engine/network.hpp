#ifndef NETWORK_H_
#define NETWORK_H_

// C/C++ headers
#include <cstddef>
#include <vector>
#include <functional>

// Engine headers
#include <activation.hpp>
#include <vector.hpp>
#include <matrix.hpp>

namespace ml {

	/*
	 * Deep Nerual Network
	 *
	 * @tparam T is the type with which calculations are performed
	 * @tparam U is the type of activation parameter scheme, ie. unary or
	 * binary
	 */
	template <class T, class U>
	class DeepNeuralNetwork {
	public:
		typedef std::pair <std::size_t, Activation<T, U>> Layer;
	private:
		std::vector <Layer>		__layers;
		std::vector <Matrix <T>>	__weights;
		std::function <T ()>		__random;
		std::size_t			__isize;
	public:
		DeepNeuralNetwork(const std::vector <Layer> &, const std::function <T ()> &);

		Vector <T> operator()(const Vector <T> &) const;

		void randomize();
	};

	template <class T, class U>
	DeepNeuralNetwork <T, U> ::DeepNeuralNetwork(const std::vector <Layer> &layers,
			const std::function <T ()> &random) : __layers(layers), __random(random),
			__isize(layers[0].first)
	{
		size_t size = __layers.size();

		for (size_t i = 0; i < size - 1; i++) {
			// Add extra column for constants (biases)
			Matrix <T> mat(__layers[i + 1].first, __layers[i].first + 1);

			__weights.push_back(mat);
		}
	}

	template <class T, class U>
	Vector <T> DeepNeuralNetwork <T, U> ::operator()(const Vector <T> &in) const
	{
		assert(in.size() == __isize);

		Vector <T> tmp = in;

		for (size_t i = 0; i < __weights.size(); i++)
			tmp = __weights[i] * tmp.append_above(T (1));
		
		return tmp;
	}

	template <class T, class U>
	void DeepNeuralNetwork <T, U> ::randomize()
	{
		using namespace std;
		for (auto &mat : __weights)
			mat.randomize(__random);
	}

}

#endif
