#ifndef NETWORK_H_
#define NETWORK_H_

// C++ Standard Libraries
#include <list>
#include <vector>

// Engine Headers
#include "functor.h"

/**
 * @brief This class represents
 * a neural network. The number
 * of layers in the network
 * (and their sizes) can be
 * specified.
 */
template <class T>
class network {
public:
	/**
	 * @brief Represents a layer
	 * of the neural network, with
	 * [size] nodes.
	 */
	struct layer {
		size_t size;
		std::vector <neuron *> lr;
	};

	/**
	 * @brief Representas a single
	 * neuron of the neural network.
	 */
	struct neuron {
		std::vector <T> weights;
		T bias;

		functor <T> *ftr;

		layer *previous;
	};
private:
	/**
	 * @brief Input vector,
	 * which is modified each
	 * time the user request
	 * an output.
	 */
	std::vector <T> ins;

	/**
	 * @brief Output vector,
	 * counterpart of the
	 * input vector of the
	 * input-ouput process
	 * of the neural network.
	 */
	std::vector <T> outs;

	functor <T> *sigmoid;

	std::list <layer> layers;
public:
	network();
	network(const std::vector <size_t> &);

	network(const std::vector <std::vector
			<std::vector <T>>> &);

	const std::vector <T> &operator()
		(const std::vector <T> &);
};

template <class T>
network <T> ::network() : sigmoid(nullptr) {}

template <class T>
network <T> ::network(const std::vector <size_t> &)
{
}

#endif
