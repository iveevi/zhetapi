#ifndef NETWORK_CUH_
#define NETWORK_CUH_

// Enable CUDA member functions
#define ZHP_CUDA

// Engine headers
#include <network.hpp>

// CUDA headers
#include <cuda/matrix.cuh>
#include <cuda/vector.cuh>
#include <cuda/activation.cuh>
#include <cuda/optimizer.cuh>

namespace zhetapi {

	namespace ml {
		// Miscellaneous functions
		template <class T>
		__host__ __device__
		Matrix <T> *NeuralNetwork <T> ::adjusted(T mu)
		{
			Matrix <T> *theta = new Matrix <T> [__size - 1];
			for (int i = 0; i < __size - 1; i++)
				theta[i] = __weights[i];

			for (int i = 0; i < __size - 1; i++)
				theta[i] += mu * __momentum[i];

			return theta;
		}
	
		// Cuda computation functions
		template <class T>
		__host__ __device__
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
				Vector <T> *a,
				Vector <T> *z) const
		{

#ifndef __CUDA_ARCH__

			if (in.size() != __isize)
				throw bad_io_dimensions();

#endif

			Vector <T> prv = in;
			Vector <T> tmp = in;

			size_t i = 0;
			while (i < __size - 1) {
				a[i] = tmp.append_above(T (1));

				prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));

				tmp = (*__layers[i + 1].second)(prv);

				Activation <T> *act = __layers[i + 1].second->derivative();

				z[i++] = (*act)(prv);

				delete act;
			}

			a[i] = tmp;
			
			return tmp;
		}

		template <class T>
		__host__ __device__
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
				Matrix <T> *weights,
				Vector <T> *a,
				Vector <T> *z) const
		{

#ifndef __CUDA_ARCH__

			if (in.size() != __isize)
				throw bad_io_dimensions();

#endif

			Vector <T> prv = in;
			Vector <T> tmp = in;

			size_t i = 0;
			while (i < __size - 1) {
				a[i] = tmp.append_above(T (1));

				prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

				tmp = (*__layers[i + 1].second)(prv);

				Activation <T> *act = __layers[i + 1].second->derivative();

				z[i++] = (*act)(prv);

				delete act;
			}

			a[i] = tmp;
			
			return tmp;
		}

		// Cuda gradient functions
		template <class T>
		__host__ __device__
		Matrix <T> *NeuralNetwork <T> ::gradient(Matrix <T> *weights,
				Vector <T> *a,
				Vector <T> *z,
				const Vector <T> &in,
				const Vector <T> &out,
				Optimizer <T> *opt,
				bool check)
		{

#ifndef __CUDA_ARCH__

			// Check dimensions of input and output
			if ((in.size() != __isize) || (out.size() != __osize))
				throw bad_io_dimensions();

#endif
		
			// Compute the actual value
			Vector <T> actual = compute(in, weights, a, z);
			
			// Get the derivative of the cost
			Optimizer <T> *dopt = opt->derivative();
			
			// Construction the Jacobian using backpropogation
			Matrix <T> *J = new Matrix <T> [__size - 1];

			Vector <T> delta = (*dopt)(out, actual);
			for (int i = __size - 2; i >= 0; i--) {
				if (i < __size - 2) {
					delta = weights[i + 1].transpose() * delta;
					delta = delta.remove_top();
				}
				
				delta = shur(delta, z[i]);

				Matrix <T> Ji = delta * a[i].transpose();

				J[i] = Ji;
			}

			// Free resources
			delete dopt;

			/* Support gradient checking only for when the function
			 * is called from the host.
			 */

#ifndef __CUDA_ARCH__

			// Skip gradient checking
			if (!check)
				return J;

			// Epsilon value
			T epsilon = 1e-8;

			// Generate individual gradients
			Matrix <T> *qJ = new Matrix <T> [__size - 1];
			for (int i = 0; i < __size - 1; i++)
				qJ[i] = weights[i];
			
			for (int i = 0; i < __size - 1; i++) {
				for (int x = 0; x < weights[i].get_rows(); x++) {
					for (int y = 0; y < weights[i].get_cols(); y++) {
						Matrix <T> *wplus = new Matrix <T> [__size - 1];
						Matrix <T> *wminus = new Matrix <T> [__size - 1];
						
						for (int k = 0; k < __size - 2; k++)
							wplus[k] = wminus[k] = weights[k];

						wplus[i][x][y] += epsilon;
						wminus[i][x][y] -= epsilon;

						Vector <T> jplus = (*opt)(out, compute_no_cache(in, wplus));
						Vector <T> jminus = (*opt)(out, compute_no_cache(in, wminus));

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

#endif

			// Return the gradient
			return J;
		}


		// Cuda training algorithm
		template <class T>
		__global__ 
		void train(NeuralNetwork <T> &net,
				typename NeuralNetwork <T> ::TrainingStatistics *ts,
				Vector <T> *ins,
				Vector <T> *outs,
				size_t size,
				Matrix <T> *Javg,
				Matrix <T> **grads,
				double *grid_opt,
				int *grid_pass)
		{
			// Block shared memoriy
			__shared__ double *block_opt;
			__shared__ int *block_pass;

			int threads = blockDim.x * gridDim.x;

			int tid = threadIdx.x + blockIdx.x + blockDim.x;
			
			int tpb = blockDim.x;
			if (threadIdx.x == 0) {
				block_opt = new double[tpb];
				block_pass = new int[tpb];
			}

			__syncthreads();

			block_opt[tid] = block_pass[tid] = 0;

			Vector <T> *aloc = new Vector <T> [net.__size];
			Vector <T> *zloc = new Vector <T> [net.__size - 1];

			for (int i = tid; i < size; i += threads) {
				Vector <T> actual = net.compute(ins[i], aloc, zloc);

				if (net.__cmp(actual, outs[i]))
					block_pass[tid]++;

				grads[i] = net.gradient(net.adjusted(0.7), aloc,
						zloc, ins[i], outs[i], net.__cost);
				
				block_opt[tid] += (*(net.__cost))(outs[i], actual)[0];
			}

			__syncthreads();

			// Add together all the BLOCK statistic
			if (threadIdx.x == 0) {
				for (int i = 0; i < tpb; i++) {
					grid_opt[blockIdx.x] += block_opt[i];
					grid_pass[blockIdx.x] += block_pass[i];
				}
			}

			// Add together all the GRID statistics 
			if (tid == 0) {
				ts->__passed = 0;
				ts->__cost = 0;
				ts->__time = 0;

				for (int i = 0; i < blockDim.x; i++) {
					ts->__passed += grid_pass[i];
					ts->__cost += grid_opt[i];
				}

				for (int i = 0; i < net.__size - 1; i++)
					Javg[i] = grads[0][i];

				for (int k = 1; k < size; k++) {
					for (int i = 0; i < net.__size - 1; i++)
						Javg[i] += grads[k][i];
				}
				
				for (int i = 0; i < net.__size - 1; i++)
					Javg[i] /= T(size);
			}
		}

		// Epoch training with CUDA
		template <class T>
		template <size_t blocks, size_t threads>
		typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
			::cuda_epochs(const DataSet <T> &ins,
				const DataSet <T> &outs,
				size_t iterations,
				size_t batch_size,
				T alpha,
				bool printing)
		{
			using namespace std;

			cout << "ins: " << ins.size() << endl;
			cout << "outs: " << outs.size() << endl;

			if (ins.size() != outs.size())
				throw bad_io_dimensions();

			std::vector <DataSet <T>> ins_batched = split(ins, batch_size);
			std::vector <DataSet <T>> outs_batched = split(outs, batch_size);

			T lr = alpha;

			T t_err = 0;
			double t_time = 0;
			size_t t_passed = 0;
			
			size_t total = 0;
			size_t passed;
			double t;
			T err;
			for (int i = 0; i < iterations; i++) {
				std::cout << std::string(20, '-')
					<< std::endl
					<< "\nEpoch #" << (i + 1)
					<< " (" << lr
					<< ")\n" << std::endl;
				
				passed = 0;
				err = 0;
				t = 0;
				for (int j = 0; j < ins_batched.size(); j++) {
					TrainingStatistics result;

					Matrix <T> *Javg = new Matrix <T>
						[__size - 1];

					Vector <T> *ins_ptr = new Vector <T>
						[ins_batched[j].size()];
					Vector <T> *outs_ptr = new Vector <T>
						[ins_batched[j].size()];

					for (int k = 0; k < ins_batched[j].size();
							k++) {
						ins_ptr[k] = ins_batched[j][k];
						outs_ptr[k] = outs_batched[j][k];
					}
					
					TrainingStatistics *dev_result;
					Matrix <T> *dev_Javg;
					Matrix <T> **dev_grads;

					Vector <T> *dev_ins;
					Vector <T> *dev_outs;

					double *dev_gopts;
					int *dev_gpass;

					int size = ins_batched[j].size();

					cudaMalloc((void **) &dev_result,
							sizeof(TrainingStatistics));
					cudaMalloc((void **) &dev_Javg,
							sizeof(Matrix <T>) *
							(__size - 1));

					cudaMalloc((void **) &dev_grads,
							sizeof(Matrix <T> *) *
							size);

					// Inputs and outputs
					cudaMalloc((void **) &dev_ins,
							sizeof(Vector <T>) *
							size);
					cudaMalloc((void **) &dev_outs,
							sizeof(Vector <T>) *
							size);

					cudaMemcpy(dev_ins, ins_ptr, size,
							cudaMemcpyHostToDevice);
					cudaMemcpy(dev_outs, outs_ptr, size,
							cudaMemcpyHostToDevice);

					// For device
					cudaMalloc((void **) &dev_gopts,
							sizeof(double) * size);
					cudaMalloc((void **) &dev_gpass,
							sizeof(int) * size);

					zhetapi::ml::train <<<blocks, threads>>> (*this,
							dev_result, dev_ins,
							dev_outs, size,
							dev_Javg, dev_grads,
							dev_gopts, dev_gpass);

					// Recopy values
					cudaMemcpy(&result, dev_result,
							sizeof(TrainingStatistics),
							cudaMemcpyDeviceToHost);
					
					cudaMemcpy(&Javg, dev_Javg,
							sizeof(Matrix <T>) *
							(__size - 1),
							cudaMemcpyDeviceToHost);

					// Free everything
					cudaFree(dev_result);
					cudaFree(dev_Javg);
					cudaFree(dev_grads);
					cudaFree(dev_ins);
					cudaFree(dev_outs);
					cudaFree(dev_gopts);
					cudaFree(dev_gpass);

					// Apply gradients

					// Free host pointers
					delete[] Javg;
					delete[] ins_ptr;
					delete[] outs_ptr;

					cout << "Batch #" << (j + 1) << " is done." << endl;

					passed += result.__passed;
					err += result.__cost;
					t += result.__time;

					lr = alpha * pow(0.1, (++total)/50000.0);
				}

				t_passed += passed;
				t_err += err;
				t_time += t;
				
				std::cout << "\nTotal cost:\t"
					<< err << std::endl
					<< "Total time:\t" << t/1000
					<< " ms" << std::endl
					<< "Cases passed:\t"
					<< passed
					<< "/" << ins.size() << " ("
					<< 100 * ((double) passed)/ins.size()
					<< "%)" << std::endl;
			}

			return {t_passed, t_err, t_time};
		}

	}

}

#endif
