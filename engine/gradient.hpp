#ifndef GRADIENT_H_
#define GRADIENT_H_

#ifndef __AVR	// Does not support AVR

// C/C++ headers
#include <mutex>
#include <thread>

#endif		// Does not support AVR

// Engine headers
#include "matrix.hpp"
#include "erf.hpp"
#include "vector.hpp"
#include "layer.hpp"
#include "dataset.hpp"

namespace zhetapi {

namespace ml {

template <class T>
Vector <T> simple_compute(
		Layer <T> *layers,
		size_t size,
		const Vector <T> &in)
{
	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size)
		layers[i++].forward_propogate(tmp, prv);

	return tmp;
}

template <class T>
Vector <T> simple_compute_cached(
		Layer <T> *layers,
		size_t size,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in)
{
	using namespace std;

	Vector <T> prv = in;
	Vector <T> tmp = in;

	size_t i = 0;
	while (i < size) {
		a[i] = tmp.append_above(T (1));

		/* cout << "tmp: " << "\n\trows = "
			<< tmp.get_rows() << ", cols = "
			<< tmp.get_cols() << endl; */

		layers[i].forward_propogate(tmp, prv);

		/* cout << "prv = " << prv << "\n\trows = "
			<< prv.get_rows() << ", cols = "
			<< prv.get_cols() << endl; */

		z[i++] = layers[i]._dact->compute(prv);
	}

	a[i] = tmp;

	return tmp;
}

// Jacobian during the processing of input to output
template <class T>
Matrix <T> *jacobian_kernel(
		Layer <T> *layers,
		size_t size,
		size_t osize,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in)
{
	// Compute the actual value
	Vector <T> actual = simple_compute_cached(layers, size, a, z, in);

	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [size];

	Vector <T> delta(osize, 1);

	for (int i = size - 1; i >= 0; i--) {
		if (i < size - 1) {
			/* delta = AVR_SWITCH(
				vvt_mult(delta, a[i]),
				std::move(rmt_and_mult(layers[i + 1]._mat, delta))
			); */
			delta = std::move(rmt_and_mult(layers[i + 1]._mat, delta));
		}

		delta.stable_shur(z[i]);

		/* J[i] = AVR_SWITCH(
			vvt_mult(delta, a[i]),
			std::move(vvt_mult(delta, a[i]))
		); */
		J[i] = std::move(vvt_mult(delta, a[i]));
	}

	// Return the gradient
	return J;
}

// Jacobian during the processing of input to output
template <class T>
Matrix <T> *jacobian_kernel(
		Layer <T> *layers,
		size_t size,
		size_t osize,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		Vector <T> &delta)
{
	// Compute the actual value
	Vector <T> actual = simple_compute_cached(layers, size, a, z, in);

	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [size];

	for (int i = size - 1; i >= 0; i--) {
		// std::cout << std::string(50, '#') << std::endl;
		// std::cout << "LOOP: " << i << std::endl;
		if (i < size - 1) {
			/* delta = AVR_SWITCH(
				rmt_and_mult(layers[i + 1]._mat, delta),
				std::move(rmt_and_mult(layers[i + 1]._mat, delta))
			); */
			/* std::cout << "alt comp delta =\t"
				<< layers[i + 1]._mat.transpose() << "\ndelta = \t\t"
				<< delta << std::endl; */

			/* std::cout << "delta-fake = \t\t" << Vector <T> (layers[i + 1]._mat.transpose() * delta).remove_top() << std::endl;
			std::cout << "delta-post =\t\t" << rmt_and_mult(layers[i + 1]._mat, delta) << std::endl; */
			delta = rmt_and_mult(layers[i + 1]._mat, delta);
			// delta = Vector <T> (layers[i + 1]._mat.transpose() * delta).remove_top();
		}

		delta.stable_shur(z[i]);

		/* std::cout << "delta = " << delta << std::endl;
		std::cout << "a[i] = " << a[i] << std::endl;
		std::cout << "a[i]^T = " << a[i].transpose() << std::endl; 8/

		/* J[i] = AVR_SWITCH(
			vvt_mult(delta, a[i]),
			std::move(vvt_mult(delta, a[i]))
		); */
		// J[i] = delta * a[i].transpose();
		J[i] = vvt_mult(delta, a[i]);

		/* std::cout << "delta * a^T =\t\t" << delta * a[i].transpose() << std::endl;
		std::cout << "J[i] =\t\t\t" << J[i] << std::endl; */
	}
	// std::cout << std::string(50, '#') << std::endl;

	// Return the gradient
	return J;
}

template <class T>
Matrix <T> *jacobian_kernel_check(
		Layer <T> *layers,
		size_t size,
		const Vector <T> &in,
		const Vector <T> &target,
		Erf <T> *erf)
{
	// Epsilon
	static T epsilon = 1e-10;

	// Making a copy of the layers
	Layer <T> *copy = new Layer <T> [size];

	for (size_t i = 0; i < size; i++)
		copy[i] = layers[i];

	Matrix <T> *J = new Matrix <T> [size];

	using namespace std;
	for (size_t i = 0; i < size; i++) {
		size_t rows = layers[i]._mat.get_rows();
		size_t cols = layers[i]._mat.get_cols();
		/* cout << "Processing index = " << i << endl;
		cout << "\trow = " << rows << endl;
		cout << "\tcols = " << cols << endl; */

		J[i] = Matrix <T> (rows, cols);
		// for (size_t)

		for (size_t x = 0; x < rows; x++) {
			for (size_t y = 0; y < cols; y++) {
				// Modify the correct layer for forward
				copy[i]._mat = layers[i]._mat;
				copy[i]._mat[x][y] += epsilon;

				Vector <T> forward = simple_compute(copy, size, in);

				// Modify the correct layer for forward
				copy[i]._mat = layers[i]._mat;
				copy[i]._mat[x][y] -= epsilon;

				Vector <T> backward = simple_compute(copy, size, in);

				T ferf = (erf->compute(forward, target))[0];
				T berf = (erf->compute(backward, target))[0];

				J[i][x][y] = (ferf - berf)/(2 * epsilon);
			}
		}
	}

	return J;
}

template <class T>
Matrix <T> *simple_gradient(
		Layer <T> *layers,
		size_t size,
		Vector <T> *a,
		Vector <T> *z,
		const Vector <T> &in,
		const Vector <T> &out,
		Erf <T> *cost)
{
	// Compute the actual value
	Vector <T> actual = simple_compute_cached(layers, size, a, z, in);

	// Get the derivative of the cost
	Erf <T> *dcost = cost->derivative();

	// Construction the Jacobian using backpropogation
	Matrix <T> *J = new Matrix <T> [size];

	Vector <T> delta = dcost->compute(out, actual);

	/* using namespace std;
	cout << "delta = " << delta << endl;
	cout << "\trows = " << delta.get_rows() << ", cols = " << delta.get_cols() << endl; */

	for (int i = size - 1; i >= 0; i--) {
		if (i < size - 1) {
			delta = AVR_SWITCH(
				rmt_and_mult(layers[i + 1]._mat, delta),
				std::move(rmt_and_mult(layers[i + 1]._mat, delta))
			);
		}

		/* std::cout << "Index = " << i  << ", size = " << size << std::endl;
		std::cout << "\tdelta = " << delta << std::endl;
		std::cout << "\tz[i] = " << z[i] << std::endl;
		cout << "\t\trows = " << z[i].get_rows() << ", cols = " << z[i].get_cols() << endl; */

		delta.stable_shur(z[i]);

		J[i] = AVR_SWITCH(
			vvt_mult(delta, a[i]),
			std::move(vvt_mult(delta, a[i]))
		);
	}

	// Free resources
	delete dcost;

	// Return the gradient
	return J;
}

#ifndef __AVR	// Does not support AVR

// TODO: Should there be an alternative for DataSet?
template <class T>
Matrix <T> *simple_batch_gradient(
		Layer <T> *layers,
		size_t size,
		Vector <T> *a,
		Vector <T> *z,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		Erf <T> *cost)
{
	size_t ds = ins.size();

	Matrix <T> *J = simple_gradient(
			layers,
			size,
			a,
			z,
			ins[0],
			outs[0],
			cost);

	for (size_t i = 1; i < ds; i++) {
		Matrix <T> *Q = nullptr;

		Q = simple_gradient(layers, size, a,
					z, ins[i], outs[i], cost);

		for (size_t k = 0; k < size; k++)
			J[k] += Q[k];

		delete[] Q;
	}

	for (size_t i = 0; i < size; i++)
		J[i] /= T(ds);

	return J;
}

template <class T>
Matrix <T> *simple_multithreaded_batch_gradient(
		Layer <T> *layers,
		size_t size,
		const DataSet <T> &ins,
		const DataSet <T> &outs,
		Erf <T> *cost,
		size_t threads)
{
	size_t ds = ins.size();


	Vector <T> *ta = new Vector <T> [size + 1];
	Vector <T> *tz = new Vector <T> [size];

	Matrix <T> *J = simple_gradient(layers, size, ta,
			tz, ins[0], outs[0], cost);

	std::mutex task_mtx;
	std::mutex addm_mtx;

	size_t task = 1;
	auto ftn = [&]() {
		Vector <T> *a = new Vector <T> [size + 1];
		Vector <T> *z = new Vector <T> [size];

		Matrix <T> *Q;
		int t;

		while (true) {
			t = -1;

			task_mtx.lock();

			if (task < ds) {
				t = task;

				task++;
			}

			task_mtx.unlock();

			if (t < 0)
				break;

			Q = simple_gradient(layers, size, a,
					z, ins[t], outs[t], cost);

			addm_mtx.lock();

			for (size_t k = 0; k < size; k++)
				J[k] += Q[k];

			addm_mtx.unlock();

			delete[] Q;
		}

		delete[] a;
		delete[] z;
	};

	std::thread *army = new std::thread[threads];
	for (size_t i = 0; i < threads; i++)
		army[i] = std::thread(ftn);

	for (size_t i = 0; i < threads; i++)
		army[i].join();

	for (size_t i = 0; i < size; i++)
		J[i] /= T(ds);

	delete[] ta;
	delete[] tz;

	return J;
}

#endif		// Does not support AVR

}

}

#endif
