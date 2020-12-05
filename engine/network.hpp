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
		* Nerual Network
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
		class NeuralNetwork {
		public:
			typedef std::pair <std::size_t, Activation <T> *> Layer;
		private:
			std::vector <Layer>		__layers;
			std::vector <Matrix <T>>	__weights;
			std::vector <Matrix <T>>	__momentum;
			std::function <T ()>		__random;
			std::size_t			__isize;
			std::size_t			__osize;

			std::vector <Vector <T>>	__xs;
			std::vector <Vector <T>>	__dxs;

			std::vector <Vector <T>>	__a;
			std::vector <Vector <T>>	__z;

			std::vector <Activation <T> *>	__dacts;
		public:
			NeuralNetwork(const std::vector <Layer> &, const std::function <T ()> &);

			~NeuralNetwork();

			Vector <T> compute(const Vector <T> &);
			Vector <T> compute(const Vector <T> &, const std::vector <Matrix <T>> &);

			Vector <T> operator()(const Vector <T> &);

			void apply_gradient(const std::vector <Matrix <T>> &, T, T);

			std::vector <Matrix <T>> gradient(const Vector <T> &,
					const Vector <T> &, Optimizer <T> *, bool = false);

			std::pair <size_t, T> train(size_t, T,
					Optimizer <T> *,
					const std::vector <Vector<T>> &,
					const std::vector <Vector <T>> &,
					const std::function <bool (const Vector <T> , const Vector <T>)> &,
					bool = false);
			void epochs(size_t, size_t, T,
					Optimizer <T> *,
					const std::vector <Vector <T>> &,
					const std::vector <Vector <T>> &,
					const std::function <bool (const Vector <T> , const Vector <T>)> &,
					bool = false);

			void randomize();

			// Printing weights
			void print() const;

			class bad_gradient {};
			class bad_io_dimensions {};
		};

		/*
		 * NOTE: The pointers allocated and passed into this function
		 * should be left alone. They will be deallocated once the scope
		 * of the network object comes to its end. In other words, DO
		 * NOT FREE ACTIVATION POINTERS, and instead let the
		 * NeuralNetwork class do the work for you.
		 */
		template <class T>
		NeuralNetwork <T> ::NeuralNetwork(const std::vector <Layer> &layers,
				const std::function <T ()> &random) : __random(random),
				__isize(layers[0].first), __osize(layers[layers.size() - 1].first),
				__layers(layers)
		{
			size_t size = __layers.size();

			for (size_t i = 0; i < size - 1; i++) {
				// Add extra column for constants (biases)
				// std::cout << "Before" << std::endl;

				Matrix <T> mat(__layers[i + 1].first, __layers[i].first + 1);

				/* std::cout << "mat: " << mat << std::endl;
				std::cout << "\trows: " << mat.get_rows() << std::endl;
				std::cout << "\tcols: " << mat.get_cols() << std::endl; */

				__weights.push_back(mat);
				__momentum.push_back(mat);
				__dacts.push_back(layers[i].second->derivative());
			}
		}

		template <class T>
		NeuralNetwork <T> ::~NeuralNetwork()
		{
			for (auto layer : __layers)
				delete layer.second;
		}

		template <class T>
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in)
		{
			assert(in.size() == __isize);

			Vector <T> prv = in;
			// Vector <T> tmp = (*__layers[0].second)(prv);
			Vector <T> tmp = in;

			__xs.clear();
			__dxs.clear();

			__a.clear();
			__z.clear();

			for (size_t i = 0; i < __weights.size(); i++) {
				__xs.insert(__xs.begin(), tmp.append_above(T (1)));

				__a.push_back(tmp.append_above(T (1)));

				prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));

				__dxs.push_back(prv);

				tmp = (*__layers[i + 1].second)(prv);

				Activation <T> *act = __layers[i + 1].second->derivative();

				__z.push_back((*act)(prv));

				delete act;
			}

			__a.push_back(tmp);
			
			return tmp;
		}

		template <class T>
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in, const std::vector <Matrix <T>> &weights)
		{
			assert(in.size() == __isize);

			using namespace std;
			// cout << "--COMPUTE!!-----------------------------" << endl;

			Vector <T> prv = in;
			Vector <T> tmp = (*__layers[0].second)(prv);
			
			// cout << "tmp: " << tmp << endl;

			for (size_t i = 0; i < __weights.size(); i++) {
				auto app = tmp.append_above(T (1));

				/* cout << "app: " << app << endl;
				cout << "weights[i]: " << weights[i] << endl; */

				prv = weights[i] * Matrix <T> (app);

				// cout << "prv: " << prv << endl;

				tmp = (*__layers[i + 1].second)(prv);				
			}

			// cout << "tmp: " << tmp << endl;
			
			return tmp;
		}

		template <class T>
		Vector <T> NeuralNetwork <T> ::operator()(const Vector <T> &in)
		{
			return compute(in);
		}

		template <class T>
		void NeuralNetwork <T> ::apply_gradient(const std::vector <Matrix <T>> &grad,
				T alpha, T mu)
		{
			assert(__weights.size() == grad.size());
			for (int i = 0; i < __weights.size(); i++) {
				// __momentum[i] = mu * __momentum[i] - alpha * grad[i];
				// __weights[i] += __momentum[i];
				__weights[i] -= alpha * grad[i];
			}
		}
			
		template <class T>
		std::vector <Matrix <T>> NeuralNetwork <T> ::gradient(const Vector <T> &in,
				const Vector <T> &out, Optimizer <T> *opt, bool check)
		{
			// Check dimensions of input and output
			if ((in.size() != __isize) || (out.size() != __osize))
				throw bad_io_dimensions();
		
			// Compute the actual value
			Vector <T> actual = (*this)(in);
			
			// Get the derivative of the cost
			Optimizer <T> *dopt = opt->derivative();
			
			// Construction the Jacobian using backpropogation
			std::vector <Matrix <T>> J;

			Vector <T> delta = (*dopt)(out, actual);
			for (int i = __weights.size() - 1; i >= 0; i--) {
				if (i < __weights.size() - 1) {
					delta = __weights[i + 1].transpose() * delta;
					delta = delta.remove_top();
				}
				
				delta = shur(delta, __z[i]);

				Matrix <T> Ji = delta * __a[i].transpose();

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
						std::vector <Matrix <T>> wplus = __weights;
						std::vector <Matrix <T>> wminus = __weights;

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
		}

		template <class T>
		std::pair <size_t, T> NeuralNetwork <T> ::train(size_t id, T alpha,
				Optimizer <T> *opt,
				const std::vector <Vector<T>> &ins,
				const std::vector <Vector<T>> &outs,
				const std::function <bool (const Vector <T>, const Vector <T>)> &crit,
				bool print)
		{
			assert(ins.size() == outs.size());
			
			using namespace std;

			const int len = 15;
			
			if (print)
				cout << "Batch #" << id << " (" << ins.size() << " samples)\t[";

			int passed = 0;
			int bars = 0;

			double opt_error = 0;
			double per_error = 0;

			std::vector <std::vector <Matrix <T>>> grads;

			int size = ins.size();
			for (int i = 0; i < size; i++) {
				Vector <T> actual = compute(ins[i]);

				if (crit(actual, outs[i]))
					passed++;

				grads.push_back(gradient(ins[i], outs[i], opt));
				
				opt_error += (*opt)(outs[i], actual)[0];
				per_error += 100 * (actual - outs[i]).norm()/outs[i].norm();

				if (print) {
					int delta = (len * (i + 1))/size;
					for (int i = 0; i < delta - bars; i++) {
						cout << "=";
						cout.flush();
					}

					bars = delta;
				}
			}

			std::vector <Matrix <T>> grad = grads[0];
			for (size_t i = 1; i < grads.size(); i++) {
				for (size_t j = 0; j < grad.size(); j++)
					grad[j] += grads[i][j];
			}
				
			for (size_t j = 0; j < grad.size(); j++)
				grad[j] /= (double) size;
			
			apply_gradient(grad, alpha, 0.7);

			if (print) {
				cout << "]\tpassed:\t" << passed << "/" << size << " ("
					<< 100 * ((double) passed)/size << "%)\t"
					<< "average error:\t" << per_error/size << "%"
					<< endl;
			}

			return std::make_pair(passed, opt_error);
		}

		template <class T>
		void NeuralNetwork <T> ::epochs(size_t runs, size_t batch, T alpha,
				Optimizer <T> *opt,
				const std::vector <Vector<T>> &ins,
				const std::vector <Vector<T>> &outs, const
				std::function<bool (const Vector <T>, const Vector <T>)> &crit,
				bool dprint)
		{
			assert(ins.size() == outs.size());

			using namespace std;

			std::vector <std::vector <Vector <T>>> ins_batched;
			std::vector <std::vector <Vector <T>>> outs_batched;

			size_t batches = 0;

			std::vector <Vector <T>> in_batch;
			std::vector <Vector <T>> out_batch;
			for (int i = 0; i < ins.size(); i++) {
				in_batch.push_back(ins[i]);
				out_batch.push_back(outs[i]);

				if (i % batch == batch - 1 || i == ins.size() - 1) {
					ins_batched.push_back(in_batch);
					outs_batched.push_back(out_batch);

					in_batch.clear();
					out_batch.clear();
				}
			}

			T perr = 0;
			T lr = alpha;

			T thresh = 0.1;

			size_t passed;
			for (int i = 0; i < runs; i++) {
				if (dprint) {
					::std::cout << ::std::string(20, '-') << ::std::endl;
					::std::cout << "Epoch #" << (i + 1) << " (" << lr << ")\n" << ::std::endl;
				}

				T err = 0;

				passed = 0;
				for (int i = 0; i < ins_batched.size(); i++) {
					auto result = train(i + 1, lr, opt, ins_batched[i], outs_batched[i], crit, dprint);

					passed += result.first;

					err += result.second;
				}

				cout << "Total Error: " << err << endl;

				if (fabs(err - perr) < thresh) {
					lr *= 0.99;

					if (thresh > 0.00001)
						thresh /= 10;
				} else if (fabs(err - perr) > 2 * thresh) {
					lr /= 0.93;

					if (thresh < 0.001)
						thresh *= 10;
				}

				perr = err;
				
				if (dprint) {
					cout << "\nCases passed:\t" << passed << "/" << ins.size() << " (" << 100 * ((double) passed)/ins.size() << "%)" << endl;
				}

				// print();
			}
		}

		template <class T>
		void NeuralNetwork <T> ::randomize()
		{
			for (auto &mat : __weights)
				mat.randomize(__random);
		}

		template <class T>
		void NeuralNetwork <T> ::print() const
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
