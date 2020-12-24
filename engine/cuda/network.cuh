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
		
		template <class T>
		__host__ __device__
		Matrix <T> *NeuralNetwork <T> ::adjusted(Matrix <T> *weights,
				Matrix <T> *momentum, size_t size, T mu)
		{
			Matrix <T> *theta = new Matrix <T> [size - 1];
			for (int i = 0; i < size - 1; i++)
				theta[i] = weights[i];

			for (int i = 0; i < size - 1; i++)
				theta[i] += mu * momentum[i];

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
		
		template <class T>
		__host__ __device__
		Vector <T> NeuralNetwork <T> ::compute_isolated(const Vector <T> &in,
				Matrix <T> *weights,
				Activation <T> **acts,
				Vector <T> *a,
				Vector <T> *z,
				size_t size)
		{
			Vector <T> prv = in;
			Vector <T> tmp = in;

			printf("Computing (Isolated)...\n");

			size_t i = 0;
			while (i < size - 1) {
				a[i] = tmp.append_above(T (1));

				prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

				Activation <T> *dev_act = copy(acts[i + 1]);

				tmp = (*dev_act)(prv);

				/* Activation <T> *dact = acts[i + 1]->derivative();

				z[i++] = (*dact)(prv);

				delete dact;
				delete dev_act; */
				i++;
			}

			printf("Done computing.\n");

			a[i] = tmp;
			
			return tmp;
		}
		
		template <class T>
		__host__ __device__
		Vector <T> NeuralNetwork <T> ::compute_no_cache_isolated(const Vector <T> &in,
				Matrix <T> *weights,
				Activation <T> **acts,
				size_t size)
		{
			Vector <T> prv = in;
			Vector <T> tmp = in;

			size_t i = 0;
			while (i < size - 1) {
				prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

				tmp = (*acts[i + 1])(prv);
			}
			
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
		
		template <class T>
		__host__ __device__
		Matrix <T> *NeuralNetwork <T> ::gradient_isolated(Matrix <T> *weights,
				Activation <T> **acts,
				size_t size,
				const Vector <T> &in,
				const Vector <T> &out,
				Optimizer <T> *opt)
		{
			// Allocate memory for a and z
			Vector <T> *a = new Vector <T> [size];
			Vector <T> *z = new Vector <T> [size - 1];

			// Compute the actual value
			Vector <T> actual = compute_isolated(in, weights, acts,
					a, z, size);
			
			// Get the derivative of the cost
			Optimizer <T> *dopt = opt->derivative();
			
			// Construction the Jacobian using backpropogation
			Matrix <T> *J = new Matrix <T> [size - 1];

			Vector <T> delta = (*dopt)(out, actual);
			for (int i = size - 2; i >= 0; i--) {
				if (i < size - 2) {
					delta = weights[i + 1].transpose() * delta;
					delta = delta.remove_top();
				}
				
				delta = shur(delta, z[i]);

				Matrix <T> Ji = delta * a[i].transpose();

				J[i] = Ji;
			}

			// Free resources
			delete[] a;
			delete[] z;

			delete dopt;

			// Return the gradient (skip checking)
			return J;
		}

		// Cuda training algorithm
		template <class T>
		__global__ 
		void training_kernel(
				Matrix <T> *weights,
				Matrix <T> *momentum,
				Activation <T> **acts,
				Optimizer <T> *opt,
				Comparator <T> cmp,
				size_t net_size,
				Vector <T> *ins,
				Vector <T> *outs,
				Matrix <T> **grads,
				double *grid_opt,
				int *grid_pass,
				size_t size,
				typename NeuralNetwork <T> ::TrainingStatistics *ts,
				Matrix <T> *Javg
			)
		{
			// Total number of threads
			int threads = blockDim.x * gridDim.x;
			
			// Block shared memory
			__shared__ double *block_opt;
			__shared__ int *block_pass;

			int tid = threadIdx.x + blockIdx.x * blockDim.x;
			
			int threads_per_block = blockDim.x;
			if (threadIdx.x == 0) {
				block_opt = new double[threads_per_block];
				block_pass = new int[threads_per_block];
			}

			__syncthreads();

			block_opt[threadIdx.x] = 0;
			block_pass[threadIdx.x] = 0;

			Vector <T> *aloc = new Vector <T> [net_size];
			Vector <T> *zloc = new Vector <T> [net_size - 1];

			for (int i = tid; i < size; i += threads) {
				Vector <T> actual = NeuralNetwork <T>
					::compute_isolated(
							ins[i],
							weights,
							acts,
							aloc,
							zloc,
							net_size);

				/* if (cmp(actual, outs[i]))
					block_pass[tid]++;

				grads[i] = NeuralNetwork <T>
					::gradient_isolated(NeuralNetwork <T>
						::adjusted(weights, momentum,
							net_size, 0.7), acts, size,
						ins[i], outs[i], opt);
				
				block_opt[tid] += (*(opt))(outs[i], actual)[0];
				*/
			}

			delete[] aloc;
			delete[] zloc;

			__syncthreads();

			// Add together all the BLOCK statistic
			if (threadIdx.x == 0) {
				for (int i = 0; i < threads_per_block; i++) {
					grid_opt[blockIdx.x] += block_opt[i];
					grid_pass[blockIdx.x] += block_pass[i];
				}
			}

			// Add together all the GRID statistics 
			if (tid == 0) {
				ts->__passed = 10;
				ts->__cost = 20;
				ts->__time = 100;

				/* for (int i = 0; i < blockDim.x; i++) {
					ts->__passed += grid_pass[i];
					ts->__cost += grid_opt[i];
				}

				for (int i = 0; i < net_size - 1; i++)
					Javg[i].stable_transfer(grads[0][i]);

				for (int k = 1; k < size; k++) {
					for (int i = 0; i < net_size - 1; i++)
						Javg[i] += grads[k][i];
				}
				
				for (int i = 0; i < net_size - 1; i++)
					Javg[i] /= T(size); */
			}

			printf("Done running kernel on batch.\n");
		}

		// Training a batch with CUDA
		template <class T>
		template <size_t blocks, size_t threads>
		typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
			::cuda_batch(const DataSet <T> &ins,
				const DataSet <T> &outs,
				T alpha,
				T mu)
		{
			// Misc variables
			size_t data_size = ins.size();
			size_t net_size = __size;

			Vector <T> *pre_ins = new Vector <T> [data_size];
			Vector <T> *pre_outs = new Vector <T> [data_size];

			for (int i = 0; i < data_size; i++) {
				pre_ins[i].copy_to_device(ins[i]);
				pre_outs[i].copy_to_device(outs[i]);
			}

			Matrix <T> *pre_weights = new Matrix <T> [net_size - 1];
			Matrix <T> *pre_momentum = new Matrix <T> [net_size - 1];
			
			for (int i = 0; i < net_size - 1; i++) {
				pre_weights[i].copy_to_device(__weights[i]);
				pre_momentum[i].copy_to_device(__momentum[i]);
			}

			Activation <T> **pre_acts = new Activation <T> *[net_size];

			for (int i = 0; i < net_size; i++) {
				cudaMalloc(&pre_acts[i], sizeof(Activation <T>));
				cudaMemcpy(pre_acts[i], __layers[i].second,
						sizeof(Activation <T>),
						cudaMemcpyHostToDevice);
			}

			Matrix <T> **pre_grads = new Matrix <T> *[data_size];

			for (int i = 0; i < data_size; i++) {
				cudaMalloc(&pre_grads[i], sizeof(Matrix <T>) * (net_size - 1));

				Matrix <T> *dev_g = new Matrix <T> [net_size - 1];
				for (int j = 0; j < net_size - 1; j++)
					dev_g[j].copy_to_device(__weights[j]);

				cudaMemcpy(pre_grads[i], dev_g,
						sizeof(Matrix <T>) * (net_size - 1),
						cudaMemcpyHostToDevice);

				delete[] dev_g;
			}

			Matrix <T> *pre_Javg = new Matrix <T> [net_size - 1];

			for (int i = 0; i < net_size - 1; i++)
				pre_Javg[i].copy_to_device(__weights[i]);

			// Storage for kernel results
			TrainingStatistics result;

			Matrix <T> *Javg = new Matrix <T> [net_size - 1];

			// Declare all the pointers
			Matrix <T> *dev_weights;
			Matrix <T> *dev_momentum;
			Activation <T> **dev_acts;
			Optimizer <T> *dev_opt;
			Comparator <T> *dev_cmp;

			Vector <T> *dev_ins;
			Vector <T> *dev_outs;

			Matrix <T> **dev_grads;
			double *dev_gopts;
			int *dev_gpass;

			TrainingStatistics *dev_result;
			Matrix <T> *dev_Javg;

			// Allocate all the pointers
			cudaMalloc(&dev_weights, sizeof(Matrix <T>) * (net_size - 1));
			cudaMalloc(&dev_momentum, sizeof(Matrix <T>) * (net_size - 1));
			cudaMalloc(&dev_acts, sizeof(Activation <T> *) * net_size);
			cudaMalloc(&dev_opt, sizeof(Optimizer <T>));
			cudaMalloc(&dev_cmp, sizeof(Comparator <T>));
			
			cudaMalloc(&dev_ins, sizeof(Vector <T>) * data_size);
			cudaMalloc(&dev_outs, sizeof(Vector <T>) * data_size);
			
			cudaMalloc(&dev_gopts, sizeof(double) * data_size);
			cudaMalloc(&dev_gpass, sizeof(int) * data_size);
			cudaMalloc(&dev_grads, sizeof(Matrix <T> *) * data_size);

			cudaMalloc(&dev_result, sizeof(TrainingStatistics));
			cudaMalloc(&dev_Javg, sizeof(Matrix <T>) * (net_size - 1));

			// Initialize the data
			cudaMemcpy(dev_weights, pre_weights, sizeof(Matrix <T>) *
					(net_size - 1), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_momentum, pre_momentum, sizeof(Matrix <T>) *
					(net_size - 1), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_acts, pre_acts, sizeof(Activation <T> *) *
					net_size, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_opt, __cost, sizeof(Optimizer <T>),
					cudaMemcpyHostToDevice);
			cudaMemcpy(dev_cmp, &__cmp, sizeof(Comparator <T>),
					cudaMemcpyHostToDevice);

			cudaMemcpy(dev_ins, pre_ins, sizeof(Vector <T>) *
					data_size, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_outs, pre_outs, sizeof(Vector <T>) *
					data_size, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_grads, pre_grads, sizeof(Matrix <T> *) *
					data_size, cudaMemcpyHostToDevice);
			
			cudaMemcpy(dev_Javg, pre_Javg, sizeof(Vector <T>) *
					(net_size - 1), cudaMemcpyHostToDevice);

			// Run the kernel
			training_kernel <<<blocks, threads>>> (
				dev_weights,
				dev_momentum,
				dev_acts,
				dev_opt,
				__cmp,
				net_size,
				dev_ins,
				dev_outs,
				dev_grads,
				dev_gopts,
				dev_gpass,
				data_size,
				dev_result,
				dev_Javg
			);

			cudaDeviceSynchronize();

			// Copy items back to host
			printf("Finished training on batch\n");

			cudaMemcpy(&result, dev_result,
					sizeof(TrainingStatistics),
					cudaMemcpyDeviceToHost);
			cudaCheckError(dev_result);

			cudaMemcpy(pre_Javg, dev_Javg, sizeof(Matrix <T>) *
					(net_size - 1), cudaMemcpyDeviceToHost);
			cudaCheckError(dev_Javg);

			// Apply the gradient
			for (int i = 0; i < net_size - 1; i++)
				pre_Javg[i].transfer_from_device(Javg[i]);

			// apply_gradient(Javg, alpha, mu);

			// Deallocate memory
			delete[] pre_ins;
			delete[] pre_outs;

			delete[] pre_weights;
			delete[] pre_momentum;

			delete[] pre_Javg;
			delete[] Javg;

			cudaFree(dev_weights);
			cudaFree(dev_momentum);
			cudaFree(dev_opt);
			cudaFree(dev_cmp);

			cudaFree(dev_result);

			cudaFree(dev_ins);
			cudaFree(dev_outs);

			cudaFree(dev_Javg);

			for (int i = 0; i < net_size; i++)
				cudaFree(pre_acts[i]);

			delete[] pre_acts;

			// Result the results
			return result;
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
					TrainingStatistics result =
						cuda_batch(ins_batched[j],
								outs_batched[j],
								lr, 0.7);

					// Save results
					std::cout << "Batch #" << (j + 1)
						<< " is done." << std::endl;

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
