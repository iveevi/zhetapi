#ifndef NETWORK_H_
#define NETWORK_H_

// C/C++ headers
#include <cstddef>
#include <vector>
#include <functional>
#include <memory>

// Engine headers
#include <activation.hpp>
#include <optimizer.hpp>
#include <vector.hpp>
#include <matrix.hpp>

namespace zhetapi {
		
	namespace ml {

		/*
		* Deep Nerual Network
		*
		* Note that layers contain pointers to activations. These are not
		* allocated and are simply COPIES of what the USER allocates. The user
		* can expect that the resources they allocated will not be destroyed
		* within this class, and that they can use the resources afterwards. In
		* other words, pointer data is READ ONLY.
		*
		* @tparam T is the type with which calculations are performed
		* @tparam U is the type of activation parameter scheme, ie. unary or
		* binary
		*/
		template <class T>
		class DeepNeuralNetwork {
		public:
			typedef std::pair <std::size_t, Activation <T> *> Layer;
		private:
			std::vector <Layer>		__layers;
			std::vector <Matrix <T>>	__weights;
			std::vector <Vector <T>>	__cache;
			std::vector <Vector <T>>	__cache_dacts;
			std::function <T ()>		__random;
			std::size_t			__isize;
			std::size_t			__osize;
		public:
			DeepNeuralNetwork(const std::vector <Layer> &, const std::function <T ()> &);

			Vector <T> operator()(const Vector <T> &);

			void learn(const Vector <T> &, const Vector <T> &, Optimizer <T> *);

			void randomize();

			// Printing weights
			void print() const;
		};

		template <class T>
		DeepNeuralNetwork <T> ::DeepNeuralNetwork(const std::vector <Layer> &layers,
				const std::function <T ()> &random) : __random(random),
				__isize(layers[0].first), __osize(layers[layers.size() - 1].first),
				__layers(layers)
		{
			size_t size = __layers.size();

			for (size_t i = 0; i < size - 1; i++) {
				// Add extra column for constants (biases)
				Matrix <T> mat(__layers[i + 1].first, __layers[i].first + 1);

				__weights.push_back(mat);
			}
		}

		template <class T>
		Vector <T> DeepNeuralNetwork <T> ::operator()(const Vector <T> &in)
		{
			assert(in.size() == __isize);

			Vector <T> prv = in;
			Vector <T> tmp = (*__layers[0].second)(prv);

			__cache.clear();
			__cache_dacts.clear();
			for (size_t i = 0; i < __weights.size(); i++) {
				__cache.push_back(tmp.append_above(T (1)));

				// std::cout << "\ttmp:\t" << tmp << std::endl;

				prv = __weights[i] * tmp.append_above(T (1));
				tmp = (*__layers[i + 1].second)(prv);
				
				__cache_dacts.push_back((*__layers[i].second->derivative())(prv));
			}
			
			return tmp;
		}

		template <class T>
		void DeepNeuralNetwork <T> ::learn(const Vector <T> &in, const Vector <T> &out, Optimizer <T> *opt)
		{
			assert(in.size() == __isize);
			assert(out.size() == __osize);

			Vector <T> tmp = (*this)(in);

			T alpha = 0.1;

			using namespace std;

			auto dO = (*(opt->derivative()))(out, tmp);

			cout << "dO: " << dO << endl;
			cout << "__cache_dacts[0] " << __cache_dacts[0] << endl;

			/* cout << "tmp:\t" << tmp << endl;
			cout << "out:\t" << out << endl;
			cout << "Error:\t" << (*opt)(out, tmp) << endl;
			cout << "DError:\t" << dO << endl;
			cout << "\trows: " << dO.get_rows() << endl;
			cout << "\tcols: " << dO.get_cols() << endl;
			cout << "DEA:\t" << shur(dO, __cache_dacts[0]) << endl;
			cout << "Cache:\t" << __cache[0] << endl;
			cout << "\trows: " << __cache[0].get_rows() << endl;
			cout << "\tcols: " << __cache[0].get_cols() << endl;
			cout << "Cache Act:\t" << __cache_dacts[0] << endl;
			cout << "\trows: " << __cache_dacts[0].get_rows() << endl;
			cout << "\tcols: " << __cache_dacts[0].get_cols() << endl; */

			dO = shur(dO, __cache_dacts[0]);

			auto gradient = dO * __cache[0].transpose();
			// cout << "Gradient:\t" << gradient << endl;

			__weights[0] -= alpha * gradient;
		}

		template <class T>
		void DeepNeuralNetwork <T> ::randomize()
		{
			for (auto &mat : __weights)
				mat.randomize(__random);
		}

		template <class T>
		void DeepNeuralNetwork <T> ::print() const
		{
			std::cout << "================================" << std::endl;
			
			std::cout << "Weights:" << std::endl;

			size_t n = 0;
			for (auto mat : __weights)
				std::cout << "[" << ++n << "]\t" << mat << std::endl;
			
			std::cout << "================================" << std::endl;
		}

	}

}

#endif
