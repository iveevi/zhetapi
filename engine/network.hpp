#ifndef NETWORK_H_
#define NETWORK_H_

// C/C++ headers
#include <cstddef>
#include <vector>
#include <functional>
#include <memory>

// Engine headers
#include <activation.hpp>
#include <optimizer.hpp>
#include <vector.hpp>
#include <matrix.hpp>

namespace ml {

	/*
	 * Deep Nerual Network
	 *
	 * Note that layers contain pointers to activations. These are not
	 * allocated and are simply COPIES of what the USER allocates. The user
	 * can expect that the resources they allocated will not be destroyed
	 * within this class, and that they can use the resources afterwards. In
	 * other words, pointer data is READ ONLY.
	 *
	 * @tparam T is the type with which calculations are performed
	 * @tparam U is the type of activation parameter scheme, ie. unary or
	 * binary
	 */
	template <class T>
	class DeepNeuralNetwork {
	public:
		typedef std::pair <std::size_t, Activation<T> *> Layer;
	private:
		std::vector <Layer>		__layers;
		std::vector <Matrix <T>>	__weights;
		std::function <T ()>		__random;
		std::size_t			__isize;
		std::size_t			__osize;
	public:
		DeepNeuralNetwork(const std::vector <Layer> &, const std::function <T ()> &);

		Vector <T> operator()(const Vector <T> &) const;

		void learn(const Vector <T> &, const Vector <T> &, Optimizer <T> *);

		void randomize();
	};

	template <class T>
	DeepNeuralNetwork <T> ::DeepNeuralNetwork(const std::vector <Layer> &layers,
			const std::function <T ()> &random) : __random(random),
			__isize(layers[0].first), __osize(layers[layers.size() - 1].first),
			__layers(layers)
	{
		size_t size = __layers.size();

		for (size_t i = 0; i < size - 1; i++) {
			// Add extra column for constants (biases)
			Matrix <T> mat(__layers[i + 1].first, __layers[i].first + 1);

			__weights.push_back(mat);
		}
	}

	template <class T>
	Vector <T> DeepNeuralNetwork <T> ::operator()(const Vector <T> &in) const
	{
		assert(in.size() == __isize);

		Vector <T> tmp = (*__layers[0].second)(in);

		for (size_t i = 0; i < __weights.size(); i++)
			tmp = (*__layers[i + 1].second)(__weights[i] * tmp.append_above(T (1)));
		
		return tmp;
	}

	template <class T>
	void DeepNeuralNetwork <T> ::learn(const Vector <T> &in, const Vector <T> &out, Optimizer <T> *opt)
	{
		assert(in.size() == __isize);
		assert(out.size() == __osize);

		Vector <T> tmp = (*this)(in);

		using namespace std;

		cout << "tmp: " << tmp << endl;
		cout << "out: " << out << endl;
		cout << "Error: " << (*opt)(out, tmp) << endl;
		cout << "DError: " << (*(opt->derivative()))(out, tmp) << endl;
	}

	template <class T>
	void DeepNeuralNetwork <T> ::randomize()
	{
		for (auto &mat : __weights)
			mat.randomize(__random);
	}

}

#endif
