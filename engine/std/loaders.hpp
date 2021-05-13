#ifndef LOADERS_H_
#define LOADERS_H_

#ifndef __AVR	// Does not support AVR

namespace zhetapi {

namespace ml {

// Forward declarations
template <class T>
class Linear;

template <class T>
class ReLU;

template <class T>
class Sigmoid;

template <class T>
class Softmax;

// Loaders
template <class T>
Activation <T> *load_linear(const std::vector <T> &args)
{
	return new Linear <T> (args[0]);
}

template <class T>
Activation <T> *load_relu(const std::vector <T> &args)
{
	return new ReLU <T> ();
}

template <class T>
Activation <T> *load_sigmoid(const std::vector <T> &args)
{
	return new Sigmoid <T> ();
}

template <class T>
Activation <T> *load_softmax(const std::vector <T> &args)
{
	return new Softmax <T> ();
}

}

}

#endif		// Does not support AVR

#endif
