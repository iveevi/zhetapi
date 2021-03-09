#include <iostream>
#include <thread>

#include <matrix.hpp>
#include <dnn.hpp>

#include <std/interval.hpp>
#include <std/linalg.hpp>

#include <std/erfs.hpp>
#include <std/activations.hpp>
#include <std/optimizers.hpp>

using namespace std;
using namespace zhetapi;

using namespace zhetapi::utility;
using namespace zhetapi::ml;
using namespace zhetapi::linalg;

#define N 5

typedef Matrix <long double> Mat;

Mat update(const Mat &H, long double dt)
{
	static int dx[] = {1, -1, 0, 0};
	static int dy[] = {0, 0, 1, -1};
	static int DIRS = 4;

	Mat copy = H;

	auto valid = [&](int i, int j) {
		if ((i < 0) || (i >= H.get_rows())
			|| (j < 0) || (j >= H.get_cols()))
			return false;
		
		return true;
	};

	// Add method for performance per cell computation
	// (Kernels that can be multithreaded, must save the
	// state of the matrix before hand)
	for (int i = 0; i < H.get_rows(); i++) {
		for (int j = 0; j < H.get_cols(); j++) {
			long double sum = H[i][j];

			for (size_t k = 0; k < DIRS; k++) {
				if (valid(i + dx[k], j + dy[k]))
					sum += dt * (H[i + dx[k]][j + dy[k]] - H[i][j]);
			}
			
			copy[i][j] = sum;
		}
	}

	return copy;
}

int main()
{
	string input;

	Erf <long double> *cost = new MeanSquaredError <long double> ();
	Optimizer <long double> *opt = new Adam <long double> ();

	DNN <long double> model(N * N, {
		Layer <long double> (2 * N * N, new ml::Sigmoid <long double> ()),
		Layer <long double> (2 * N * N, new ml::ReLU <long double> ()),
		Layer <long double> (2 * N * N, new ml::Sigmoid <long double> (), RandomInitializer <long double> (), 0.1),
		Layer <long double> (N * N, new ml::Linear <long double> ())
	});

	model.set_cost(cost);
	model.set_optimizer(opt);

	Mat heat_0(N, N,
		[&](size_t i, size_t j) {
			return 10 * (1_I).uniform();
		}
	);

	Mat heat = heat_0;

	for (size_t i = 0; i < 1000; i++) {
		// Generate the actual input
		Vec input = flatten(heat);

		// Predicted update
		cout << "Prediction:" << endl;

		Vec pred = model(input);

		pretty(cout, fold(pred, N, N)) << "\n" << endl;
		
		// Real update
		cout << "Real:" << endl;

		heat = update(heat, 0.0001);

		pretty(cout, heat) << "\n" << endl;

		cout << "Error: " << (cost->compute(pred, input))[0] << endl;
		
		cout << string(100, '=') << endl;

		model.fit(input, flatten(heat));

		this_thread::sleep_for(10ms);
	}

	cout << "Post training, starting over with same sequence:";
	getline(cin, input);

	heat = heat_0;
	
	for (size_t i = 0; i < 1000; i++) {
		// Generate the actual input
		Vec input = flatten(heat);

		// Predicted update
		cout << "Prediction:" << endl;

		Vec pred = model(input);

		pretty(cout, fold(pred, N, N)) << "\n" << endl;
		
		// Real update
		cout << "Real:" << endl;

		heat = update(heat, 0.0001);

		pretty(cout, heat) << "\n" << endl;

		cout << "Error: " << (cost->compute(pred, input))[0] << endl;
		
		cout << string(100, '=') << endl;

		this_thread::sleep_for(10ms);
	}
	
	cout << "Post training, starting over with new sequence:";
	getline(cin, input);
	
	heat = Mat(N, N,
		[&](size_t i, size_t j) {
			return 10 * (1_I).uniform();
		}
	);
	
	for (size_t i = 0; i < 1000; i++) {
		// Generate the actual input
		Vec input = flatten(heat);

		// Predicted update
		cout << "Prediction:" << endl;

		Vec pred = model(input);

		pretty(cout, fold(pred, N, N)) << "\n" << endl;
		
		// Real update
		cout << "Real:" << endl;

		heat = update(heat, 0.0001);

		pretty(cout, heat) << "\n" << endl;

		cout << "Error: " << (cost->compute(pred, input))[0] << endl;
		
		cout << string(100, '=') << endl;

		this_thread::sleep_for(10ms);
	}
}
