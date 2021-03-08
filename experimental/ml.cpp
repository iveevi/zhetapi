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
	DNN <double> model(N * N, {
		Layer <double> (2 * N * N, new ml::Sigmoid <double> ()),
		Layer <double> (2 * N * N, new ml::ReLU <double> ()),
		Layer <double> (2 * N * N, new ml::Sigmoid <double> ()),
		Layer <double> (2 * N * N, new ml::ReLU <double> ())
	});

	Mat heat(N, N,
		[&](size_t i, size_t j) {
			return (1_I).uniform();
		}
	);

	for (size_t i = 0; i < 1000; i++) {
		printf("\033[H\033[J");

		pretty(cout, heat) << endl;

		heat = update(heat, 0.0001);

		cout << "flattened: " << flatten(heat) << endl;
		cout << "fold: " << fold(flatten(heat), N, N) << endl;

		this_thread::sleep_for(10ms);
	}
}