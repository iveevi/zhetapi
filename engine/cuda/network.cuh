#ifndef NETWORK_CU_H_
#define NETWORK_CU_H_

// Enable CUDA member functions
#define ZHP_CUDA

// Engine headers
#include <network.hpp>

// CUDA headers
#include <cuda/activation.cuh>
#include <cuda/optimizer.cuh>

namespace zhetapi {

	namespace ml {
		
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
				const std::vector <Matrix <T>> &weights,
				std::vector <Vector <T>> &a,
				std::vector <Vector <T>> &z) const
		{
			if (in.size() != __isize)
				throw bad_io_dimensions();

			Vector <T> prv = in;
			Vector <T> tmp = in;

			a.clear();
			z.clear();

			for (size_t i = 0; i < __weights.size(); i++) {
				a.push_back(tmp.append_above(T (1)));

				prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

				tmp = (*__layers[i + 1].second)(prv);

				Activation <T> *act = __layers[i + 1].second->derivative();

				z.push_back((*act)(prv));

				delete act;
			}

			a.push_back(tmp);
			
			return tmp;
		}

		template <class T>
		__global__ 
		void train(const NeuralNetwork <T> &net,
				typename NeuralNetwork <T> ::TrainingStatistics *ts,
				const DataSet <T> &ins,
				const DataSet <T> &outs,
				Vector <T> **grads,
				double *optes,
				double *peres,
				double *pass)
		{
			int threads = gridDim.x * blockDim.x;
			int tid = threadIdx.x + blockIdx.x + blockDim.x;
			int size = ins.size();

			Vector <T> *aloc = new Vector <T> (net.__size);
			Vector <T> *zloc = new Vector <T> (net.__size - 1);

			for (int i = tid; i < size; i += threads) {
				Vector <T> actual = net.compute(ins[i], aloc, zloc);

				// TODO: Determine whether the the
				// following if statement is
				// hazardous.
				if (net.__cmp(actual, outs[i]))
					pass[tid]++;

				grads[i] = net.gradient(net.adjusted(0.7), aloc,
						zloc, ins[i], outs[i], net.__cost);
				
				optes[tid] += (*(net.__cost))(outs[i], actual)[0];
				peres[tid] += 100 * (actual - outs[i]).norm()/outs[i].norm();
			}
		}

	}

}

#endif
