#ifndef TRAINING_H_
#define TRAINING_H_

// Engine headers
#include <dataset.hpp>
#include <display.hpp>
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
template <class T>
struct PerformanceStatistics {
	T	__cost		= T(0);
	size_t	__passed	= 0;
	double	__kernel_time	= 0;
};

// Non-statistical methods (without performance statistics)
template <class T>
void train_mini_batch(
		NeuralNetwork <T> &net,
		const DataSet <T> &ins,
		const DataSet <T> &outs)
{
	assert(ins.size() == outs.size());

	size_t n;

	n = ins.size();
	for (size_t i = 0; i < n; i++)
		net.fit(ins[i], outs[i]);
}

template <class T>
void train_dataset(
		NeuralNetwork <T> &net,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t batch_size)
{
	assert(ins.size() == outs.size());

	std::vector <DataSet <T>> input_batches = ins.split(batch_size);
	std::vector <DataSet <T>> output_batches = outs.split(batch_size);

	size_t n;

	n = input_batches.size();
	for (size_t i = 0; i < n; i++) {
		train_mini_batch(net,
				input_batches[i],
				output_batches[i]);
	}
}

// Statistical counterparts of the above (with performance metrics)
template <class T>
PerformanceStatistics <T> train_mini_batch_perf(
		NeuralNetwork <T> &net,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		Erf <T> *cost,
		Comparator <T> cmp = __def_cmp <T>,
		Display::type display = 0)
{
	assert(ins.size() == outs.size());

	PerformanceStatistics <T> ns;
	Vector <double> to;
	T perr;
	size_t n;

	perr = 0;
	n = ins.size();

	// Performance statistics first
	for (size_t i = 0; i < n; i++) {
		to = net(ins[i]);
		ns.__cost += (*cost)(to, outs[i])[0];
		ns.__passed += (cmp(to, outs[i]));

		perr += fabs((to - outs[i]).norm() / outs[i].norm());
	}

	net.fit(ins, outs);

	perr /= n;
	if (display & Display::batch) {
		std::cout << "Batch done:"
			<< " %-err = " << 100 * perr << "%"
			<< " %-passed = " << (100.0 * ns.__passed)/n << "%"
			<< " #passed = " << ns.__passed
			<< std::endl;
	}

	return ns;
}

template <class T>
PerformanceStatistics <T> train_dataset_perf(
		NeuralNetwork <T> &net,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t batch_size,
		Erf <T> *cost,
		Comparator <T> cmp = __def_cmp <T>,
		Display::type display = 0)
{
	assert(ins.size() == outs.size());

	std::vector <DataSet <T>> input_batches = split(ins, batch_size);
	std::vector <DataSet <T>> output_batches = split(outs, batch_size);

	PerformanceStatistics <T> ns;
	PerformanceStatistics <T> bs;
	size_t n;
	
	n = input_batches.size();
	for (size_t i = 0; i < n; i++) {
		bs = train_mini_batch_perf(net,
				input_batches[i],
				output_batches[i],
				cost,
				cmp,
				display);

		ns.__cost += bs.__cost;
		ns.__passed += bs.__cost;
	}

	return ns;
}

}

}

#endif
