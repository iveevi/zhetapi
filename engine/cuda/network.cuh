#ifndef NETWORK_CU_H_
#define NETWORK_CU_H_

// Enable CUDA member functions
#define ZHP_CUDA

// Engine headers
#include <network.hpp>

namespace zhetapi {

	namespace cuda {

		__global__ void kernel(int a, int b)
		{
			a = b;
		}

		/* template <class T>
		__device__ std::vector <Matrix <T>> NeuralNetwork <T>
			::gradient(const std::vector <Matrix <T>> &weights,
				std::vector <Vector <T>> &a,
				std::vector <Vector <T>> &z,
				const Vector <T> &in,
				const Vector <T> &out,
				Optimizer <T> *opt,
				bool check)
		{
			// Check dimensions of input and output
			if ((in.size() != __isize) || (out.size() != __osize))
				throw bad_io_dimensions();
		
			// Compute the actual value
			Vector <T> actual = compute(in, weights, a, z);
			
			// Get the derivative of the cost
			Optimizer <T> *dopt = opt->derivative();
			
			// Construction the Jacobian using backpropogation
			std::vector <Matrix <T>> J;

			Vector <T> delta = (*dopt)(out, actual);
			for (int i = weights.size() - 1; i >= 0; i--) {
				if (i < weights.size() - 1) {
					delta = weights[i + 1].transpose() * delta;
					delta = delta.remove_top();
				}
				
				delta = shur(delta, z[i]);

				Matrix <T> Ji = delta * a[i].transpose();

				J.insert(J.begin(), Ji);
			}

			// Free resources
			delete dopt;

			// Skip gradient checking
			if (!check)
				return J;

			// Epsilon value
			T epsilon = 1e-8;

			// Generate individual gradients
			std::vector <Matrix <T>> qJ = __weights;
			for (int i = 0; i < __weights.size(); i++) {
				for (int x = 0; x < __weights[i].get_rows(); x++) {
					for (int y = 0; y < __weights[i].get_cols(); y++) {
						std::vector <Matrix <T>> wplus = weights;
						std::vector <Matrix <T>> wminus = weights;

						wplus[i][x][y] += epsilon;
						wminus[i][x][y] -= epsilon;

						Vector <T> jplus = (*opt)(out, compute(in, wplus));
						Vector <T> jminus = (*opt)(out, compute(in, wminus));

						qJ[i][x][y] = (jplus[0] - jminus[0])/(epsilon + epsilon);

						// Compute the error
						T a = J[i][x][y];
						T b = qJ[i][x][y];

						T d = a - b;

						T e = (d * d) / (a * a + b * b + epsilon);

						// If the error is more than epsilon throw an error
						if (e > epsilon)
							throw bad_gradient();
					}
				}

			}

			// Return the gradient
			return J;
		} */

	}

}

#endif
