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
	struct neuron;

	/**
	 * @brief Represents a layer
	 * of the neural network, with
	 * [size] nodes.
	 */
	struct layer {
		size_t size;
		std::vector <neuron *> lr;

		layer(size_t i = 0) : size(i), lr(i, nullptr) {}
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
	layer ins;

	/**
	 * @brief Output vector,
	 * counterpart of the
	 * input vector of the
	 * input-ouput process
	 * of the neural network.
	 */
	layer outs;

	/* @brief The hidden layers
	 * of the neural network.
	 * May or may not be part
	 * of the actual network, i.e.
	 * could be empty. */
	std::vector <layer> layers;
	
	/* @brief The sigmoid function
	 * for restricting the domain of
	 * the output of the neurons. */
	functor <T> *sigmoid;
public:
	network();
	network(const std::vector <size_t> &);
	network(const std::vector <std::vector <std::pair <std::vector <T>, T>>> &);

	void assign(const std::vector <std::vector
			<std::vector <T>>> &);

	void print() const;

	const std::vector <T> &operator()
		(const std::vector <T> &);
};

template <class T>
network <T> ::network() : sigmoid(nullptr), layers(0, layer(0)) {}

template <class T>
network <T> ::network(const std::vector <size_t> &fmt)
{
	assert(fmt.size() >= 2);

	ins = layer(fmt[0]);

	for (size_t i = 1; i < fmt.size() - 1; i++)
		layers.push_back(layer(fmt[i]));

	outs = layer(fmt[fmt.size() - 1]);
}

template <class T>
network <T> ::network(const std::vector <std::vector <std::pair <std::vector <T>, T>>> &fmt)
{
	assert(fmt.size() >= 2);

	ins = layer(fmt[0].size());
	for (size_t i = 0; i < ins.size; i++)
		ins.lr[i] = new neuron {{}, fmt[0][i].second, nullptr, nullptr};

	for (size_t i = 1; i < fmt.size() - 1; i++) {
		layers.push_back(layer(fmt[i].size()));

		for (size_t j = 0; j < layers[j - 1].size; j++)
			layers[i - 1].lr[j] = new neuron {fmt[i][j].first, fmt[i][j].second, nullptr, nullptr};
	}

	outs = layer(fmt[fmt.size() - 1].size());
	for (size_t i = 0; i < ins.size; i++)
		outs.lr[i] = new neuron {fmt[fmt.size() - 1][i].first, fmt[fmt.size() - 1][i].second, nullptr, nullptr};
}

template <class T>
void network <T> ::print() const
{
	size_t mx = max(ins.size, outs.size);
	for (size_t i = 0; i < layers.size(); i++)
		mx = max(mx, layers[i].size);

	cout << "IN\t\t";
	for (size_t i = 0; i < layers.size(); i++)
		cout << "L" << (i + 1) << "\t\t";
	cout << "OUT" << endl;

	cout << string(2 * 8 * (layers.size() + 2), '=') << endl;

	for (size_t i = 0; i < mx; i++) {
		if (ins.size > i)
			cout << ins.lr[i];
		cout << "\t";

		for (auto lyr : layers) {
			if (lyr.size > i)
				cout << ins.lr[i];
			cout << "\t";
		}

		if (outs.size > i)
			cout << outs.lr[i];

		cout << endl;
	}
}

#endif
