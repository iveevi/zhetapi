#ifndef NETWORK_H_
#define NETWORK_H_

// C++ Standard Libraries
#include <list>
#include <vector>

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
	struct layer {
		size_t size;
		std::vector <neuron> lr;
	};

	struct neuron {
		layer *l;
		operand <T> b;
		std::vector <operand <T>> w;
	};
private:
	operand <T> *ins;
	operand <T> *outs;

	std::list <layer> layers;
public:
	network();
	network(const std::vector <size_t> &);

	const std::vector <operand <T>> &operator()
		(const std::vector <operand <T>> &);
};

#endif
