#ifndef TRAINING_H_
#define TRAINING_H_

// Engine headers
#include <dataset.hpp>
#include <display.hpp>
#include <dnn.hpp>
#include <erf.hpp>
#include <optimizer.hpp>

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
bool _def_cmp(const Vector <T> &a, const Vector <T> &e)
{
	return a == e;
};

// Training statistics
template <class T>
struct PerformanceStatistics {
	T	_cost		= T(0);
	size_t	_passed	= 0;
	double	_kernel_time	= 0;
};

// Fitting a single I/O pair
template <class T>
void fit(
		DNN <T> &dnn,
		const Vector <T> &in,
		const Vector <T> &out,
		Erf <T> *erf,
		Optimizer <T> *opt)
{
	Erf <T> *derf = erf->derivative();

	// Use cached compute later
	Vector <T> actual = dnn(in);

	Matrix <T> *J;
	
	J = dnn.jacobian(in, derf->compute(actual, out));
	J = opt->update(J);

	dnn.apply_gradient(J);

	delete[] J;
	delete derf;
}

template <class T>
void fit(
		DNN <T> &dnn,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		Erf <T> *erf,
		Optimizer <T> *opt)
{
	/* if (ins.size() != outs.size())
		throw bad_io_dimensions();

	if ((ins[0].size() != _isize) || (outs[0].size() != _osize))
		throw bad_io_dimensions();

	if (!_opt)
		throw null_optimizer();
	
	if (!_cost)
		throw null_loss_function(); */

	Matrix <T> *J;
	
	// Put batch gradient (multithread and etc) in dnn method (batch jacobian)
	J = simple_batch_gradient(dnn.layers(), dnn.size(), dnn.acache(), dnn.zcache(), ins, outs, erf);
	J = opt->update(J, dnn.size());

	dnn.apply_gradient(J);

	delete[] J;
}

template <class T>
void multithreaded_fit(
		DNN <T> &dnn,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		Erf <T> *erf,
		Optimizer <T> *opt,
		size_t threads)
{
	/* if (ins.size() != outs.size())
		throw bad_io_dimensions();

	if ((ins[0].size() != _isize) || (outs[0].size() != _osize))
		throw bad_io_dimensions();

	if (!_opt)
		throw null_optimizer();
	
	if (!_cost)
		throw null_loss_function(); */

	Matrix <T> *J;
	
	J = simple_multithreaded_batch_gradient(
			dnn.layers(),
			dnn.size(),
			ins,
			outs,
			erf,
			threads);
	J = opt->update(J, dnn.size());

	dnn.apply_gradient(J);

	delete[] J;
}

// Non-statistical methods (without performance statistics)
template <class T>
void train_dataset(
		DNN <T> &dnn,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t batch_size,
		size_t threads = 1)
{
	assert(ins.size() == outs.size());

	std::vector <DataSet <T>> input_batches = split(ins, batch_size);
	std::vector <DataSet <T>> output_batches = split(outs, batch_size);

	size_t n;

	n = input_batches.size();
	for (size_t i = 0; i < n; i++) {
		if (threads > 1)
			dnn.multithreaded_fit(input_batches[i], output_batches[i], threads);
		else
			dnn.fit(input_batches[i], output_batches[i]);
	}
}

// Statistical counterparts of the above (with performance metrics)
template <class T>
PerformanceStatistics <T> train_mini_batch_perf(
		DNN <T> &dnn,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		Erf <T> *erf,
		Optimizer <T> *opt,
		Comparator <T> cmp = _def_cmp <T>,
		Display::type display = 0,
		size_t threads = 1)
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
		to = dnn(ins[i]);
		ns._cost += erf->compute(to, outs[i]).x();
		ns._passed += cmp(to, outs[i]);

		perr += fabs((to - outs[i]).norm() / outs[i].norm());
	}

	if (threads > 1)
		multithreaded_fit(dnn, ins, outs, erf, opt, threads);
	else
		fit(dnn, ins, outs, erf, opt);

	perr /= n;
	if (display & Display::batch) {
		std::cout << "Batch done:"
			<< " %-err = " << 100 * perr << "%"
			<< " %-passed = " << (100.0 * ns._passed)/n << "%"
			<< " #passed = " << ns._passed
			<< std::endl;
	}

	return ns;
}

template <class T>
PerformanceStatistics <T> train_dataset_perf(
		DNN <T> &dnn,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		size_t batch_size,
		Erf <T> *erf,
		Optimizer <T> *opt,
		Display::type display = 0,
		size_t threads = 1,
		Comparator <T> cmp = _def_cmp <T>)
{
	assert(ins.size() == outs.size());

	std::vector <DataSet <T>> input_batches = split(ins, batch_size);
	std::vector <DataSet <T>> output_batches = split(outs, batch_size);

	PerformanceStatistics <T> ns;
	PerformanceStatistics <T> bs;
	size_t n;
	
	n = input_batches.size();
	for (size_t i = 0; i < n; i++) {
		bs = train_mini_batch_perf(dnn,
				input_batches[i],
				output_batches[i],
				erf,
				opt,
				cmp,
				display,
				threads);

		ns._cost += bs._cost;
		ns._passed += bs._cost;
	}

	return ns;
}

}

}

#endif
