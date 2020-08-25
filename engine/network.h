#ifndef NETWORK_H_
#define NETWORK_H_

// C/C++
#include <vector>

// Engine headers
#include <activations.h>
#include <vector.h>
#include <matrix.h>

namespace ml {

	template <class T>
	class DeepNeuralNetwork {
	public:
		typedef std::pair <std::size_t, Activation<T>> Layer;
	private:
		std::vector <Layer> layers;

		std::vector <Matrix <T>> weights;
	public:
		DeepNeuralNetwork(const std::vector <Layer> &);
	};

	template <class T>
	DeepNeuralNetwork <T> ::DeepNeuralNetwork(const std::vector <Layer> &__layers) :
			layers(__layers)
	{

	}

}

#endif
