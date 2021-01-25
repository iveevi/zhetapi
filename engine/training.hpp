#ifndef TRAINING_H_
#define TRAINING_H_

// Engine headers
#include <network.hpp>

namespace zhetapi {

namespace ml {

/*
 * TODO: Abstract the training methods into a single training program class,
 * which stores the dataset(s) for its entire life. This way, training can be
 * made much faster in the GPU, by preloading the datasets into the GPU and
 * releasing them only when they are to be deconstructed.
 */

// Diagnosing function for training
template <class T>
using Comparator = bool (*)(const Vector <T> &, const Vector <T> &);

// Default diagnoser
template <class T>
bool __def_cmp(const Vector <T> &a, const Vector <T> &e)
{
	return a == e;
};

// Training statistics
struct NetworkStatistics {
	T	__cost		= T(0);
	size_t	__passed	= 0;
	double	__kernel_time	= 0;
};

// Add a quiet (void return) counterpart
template <class T>
ModelStatistics train_mini_batch(
		NeuralNetwork <T> &net,
		const DataSet <T> &ins,
		const DataSet <T> &outs
		Comparator <T> cmp = __def_cmp <T> ())
{
	assert(ins.size() == outs.size());

	NetworkStatistics ns;
	Vector <double> to;
	size_t n;

	n = ins.size();
	for (size_t i = 0; i < n; i++) {
		to = net(ins[i]);
		ns.__cost += net.__cost(to, outs[i]);
		ns.__passed += (cmp(to, outs[i]));
	}

	return ns;
}

// Add a quiet (void return) counterpart
template <class T>
ModelStatistics train_dataset(
		NeuralNetwork <T> &net,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t batch_size,
		Comparator <T> cmp = __def_cmp <T> ())
{
	assert(ins.size() == outs.size());

	std::vector <DataSet <T>> input_batches = ins.split(batch_size);
	std::vector <DataSet <T>> output_batches = outs.split(batch_size);

	NetworkStatistics ns;
	NetworkStatistics bs;
	size_t n;

	n = input_batches.size();
	for (size_t i = 0; i < n; i++) {
		bs = train_mini_batch(net,
				input_batches[i],
				output_batches[i],
				cmp);

		ns.__cost += bs.__cost;
		ns.__passed += bs.__cost;
	}
}

}

}

#endif
