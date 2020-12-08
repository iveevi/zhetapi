#ifndef NETWORK_H_
#define NETWORK_H_

// C/C++ headers
#include <chrono>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <memory>
#include <thread>
#include <vector>

// Engine headers
#include <activation.hpp>
#include <dataset.hpp>
#include <matrix.hpp>
#include <optimizer.hpp>
#include <vector.hpp>

namespace zhetapi {
		
	namespace ml {

		/*
		* Nerual Network
		*
		* @tparam T is the type with which calculations are performed
		* @tparam U is the type of activation parameter scheme, ie. unary or
		* binary
		*/
		template <class T>
		class NeuralNetwork {
		public:
			using Layer = std::pair <std::size_t, Activation <T> *>;
			using Comparator = std::function <bool (const Vector <T> , const Vector <T>)>;
			
			// Training statistics
			struct TrainingStatistics {
				size_t	__passed;
				T	__cost;
				double	__time;
			};

			// Exceptions
			class bad_gradient {};
			class bad_io_dimensions {};
		private:
			std::vector <Layer>		__layers;
			std::vector <Matrix <T>>	__weights;
			std::vector <Matrix <T>>	__momentum;
			std::function <T ()>		__random;
			size_t				__isize;
			size_t				__osize;

			std::vector <Vector <T>>	__a;
			std::vector <Vector <T>>	__z;

			Optimizer <T> *			__cost;
			Comparator			__cmp;
			
			static Comparator		__default_comparator;
		public:
			NeuralNetwork(const std::vector <Layer> &, const std::function <T ()> &);

			~NeuralNetwork();

			// Setters
			void set_cost(Optimizer <T> *);
			void set_comparator(Comparator);

			Vector <T> compute(const Vector <T> &);
			Vector <T> compute(const Vector <T> &, const std::vector <Matrix <T>> &);
			Vector <T> compute(const Vector <T> &,
					std::vector <Vector <T>> &,
					std::vector <Vector <T>> &);
			Vector <T> compute(const Vector <T> &,
					const std::vector <Matrix <T>> &,
					std::vector <Vector <T>> &,
					std::vector <Vector <T>> &);

			Vector <T> operator()(const Vector <T> &);

			void apply_gradient(const std::vector <Matrix <T>> &, T, T);

			std::vector <Matrix <T>> adjusted(T mu);

			std::vector <Matrix <T>> gradient(const Vector <T> &,
					const Vector <T> &,
					Optimizer <T> *,
					bool = false);
			std::vector <Matrix <T>> gradient(const std::vector <Matrix <T>> &,
					const Vector <T> &,
					const Vector <T> &,
					Optimizer <T> *,
					bool = false);
			
			std::vector <Matrix <T>> gradient(const std::vector <Matrix <T>> &,
					std::vector <Vector <T>> &,
					std::vector <Vector <T>> &,
					const Vector <T> &,
					const Vector <T> &,
					Optimizer <T> *,
					bool = false);

#ifdef ZHP_CUDA
			__device__ std::vector <Matrix <T>> gradient(const std::vector <Matrix <T>> &,
				std::vector <Vector <T>> &,
				std::vector <Vector <T>> &,
				const Vector <T> &,
				const Vector <T> &,
				Optimizer <T> *,
				bool = false);
#endif

			template <size_t = 1>
			TrainingStatistics train(const DataSet <T> &,
				const DataSet <T> &,
				T, size_t = 0,
				bool = false);
		
			template <size_t = 1>
			TrainingStatistics epochs(const DataSet <T> &,
				const DataSet <T> &,
				size_t, size_t, T,
				bool = false);

			void randomize();

			// Printing weights
			void print() const;
		};

		// Static variables
		template <class T>
		bool default_comparator(Vector <T> a, Vector <T> e)
		{
			return a == e;
		};

		template <class T>
		typename NeuralNetwork <T> ::Comparator
		NeuralNetwork <T> ::__default_comparator = default_comparator <T>;

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
				__layers(layers), __cost(nullptr), __cmp(__default_comparator)
		{
			size_t size = __layers.size();

			for (size_t i = 0; i < size - 1; i++) {
				// Add extra column for constants (biases)
				Matrix <T> mat(__layers[i + 1].first, __layers[i].first + 1);

				__weights.push_back(mat);
				__momentum.push_back(mat);
			}
		}

		template <class T>
		NeuralNetwork <T> ::~NeuralNetwork()
		{
			for (auto layer : __layers)
				delete layer.second;
		}

		// Setters
		template <class T>
		void NeuralNetwork <T> ::set_cost(Optimizer<T> *opt)
		{
			__cost = opt;
		}

		template <class T>
		void NeuralNetwork <T> ::set_comparator(Comparator cmp)
		{
			__cmp = cmp;
		}

		template <class T>
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in)
		{
			if (in.size() != __isize)
				throw bad_io_dimensions();

			Vector <T> prv = in;
			Vector <T> tmp = in;

			__a.clear();
			__z.clear();

			for (size_t i = 0; i < __weights.size(); i++) {
				__a.push_back(tmp.append_above(T (1)));

				prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));

				tmp = (*__layers[i + 1].second)(prv);

				Activation <T> *act = __layers[i + 1].second->derivative();

				__z.push_back((*act)(prv));

				delete act;
			}

			__a.push_back(tmp);
			
			return tmp;
		}
		
		template <class T>
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
				std::vector <Vector <T>> &a,
				std::vector <Vector <T>> &z)
		{
			if (in.size() != __isize)
				throw bad_io_dimensions();

			Vector <T> prv = in;
			Vector <T> tmp = in;

			a.clear();
			z.clear();

			for (size_t i = 0; i < __weights.size(); i++) {
				a.push_back(tmp.append_above(T (1)));

				prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));

				tmp = (*__layers[i + 1].second)(prv);

				Activation <T> *act = __layers[i + 1].second->derivative();

				z.push_back((*act)(prv));

				delete act;
			}

			a.push_back(tmp);
			
			return tmp;
		}

		template <class T>
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in, const std::vector <Matrix <T>> &weights)
		{
			if (in.size() != __isize)
				throw bad_io_dimensions();

			Vector <T> prv = in;
			Vector <T> tmp = in;

			__a.clear();
			__z.clear();

			for (size_t i = 0; i < __weights.size(); i++) {
				__a.push_back(tmp.append_above(T (1)));

				prv = weights[i] * Matrix <T> (tmp.append_above(T (1)));

				tmp = (*__layers[i + 1].second)(prv);

				Activation <T> *act = __layers[i + 1].second->derivative();

				__z.push_back((*act)(prv));

				delete act;
			}

			__a.push_back(tmp);
			
			return tmp;
		}
		
		template <class T>
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in,
				const std::vector <Matrix <T>> &weights,
				std::vector <Vector <T>> &a,
				std::vector <Vector <T>> &z)
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
		Vector <T> NeuralNetwork <T> ::operator()(const Vector <T> &in)
		{
			return compute(in);
		}

		template <class T>
		void NeuralNetwork <T> ::apply_gradient(const std::vector <Matrix <T>> &grad,
				T alpha,
				T mu)
		{
			for (int i = 0; i < __weights.size(); i++) {
				__momentum[i] = mu * __momentum[i] - alpha * grad[i];
				__weights[i] += __momentum[i];
			}
		}

		template <class T>
		std::vector <Matrix <T>> NeuralNetwork <T> ::adjusted(T mu)
		{
			std::vector <Matrix <T>> theta = __weights;
			for (int i = 0; i < __weights.size(); i++)
				theta[i] += mu * __momentum[i];

			return theta;
		}
			
		template <class T>
		std::vector <Matrix <T>> NeuralNetwork <T> ::gradient(const Vector <T> &in,
				const Vector <T> &out,
				Optimizer <T> *opt,
				bool check)
		{
			// Check dimensions of input and output
			if ((in.size() != __isize) || (out.size() != __osize))
				throw bad_io_dimensions();
		
			// Compute the actual value
			Vector <T> actual = compute(in);
			
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
		std::vector <Matrix <T>> NeuralNetwork <T> ::gradient(const std::vector <Matrix <T>> &weights,
				const Vector <T> &in,
				const Vector <T> &out,
				Optimizer <T> *opt,
				bool check)
		{
			// Check dimensions of input and output
			if ((in.size() != __isize) || (out.size() != __osize))
				throw bad_io_dimensions();
		
			// Compute the actual value
			Vector <T> actual = compute(in, weights);
			
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
		}
		
		template <class T>
		std::vector <Matrix <T>> NeuralNetwork <T> ::gradient(const std::vector <Matrix <T>> &weights,
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
		}

		template <class T>
		template <size_t threads>
		typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
			::train(const DataSet <T> &ins,
				const DataSet <T> &outs,
				T alpha,
				size_t id,
				bool printing)
		{
			if (ins.size() != outs.size())
				throw bad_io_dimensions();
			
			const int len = 15;
			const int width = 7;
		
			std::chrono::high_resolution_clock::time_point start;
			std::chrono::high_resolution_clock::time_point end;
			std::chrono::high_resolution_clock::time_point total;
			
			std::chrono::high_resolution_clock clk;

			if (printing) {
				std::string str = "#" + std::to_string(id);

				std::cout << "Batch " << std::setw(6)
					<< str << " (" << ins.size() << ")";
			}

			size_t passed = 0;

			int bars = 0;

			double opt_error = 0;
			double per_error = 0;

			start = clk.now();

			int size = ins.size();

			std::vector <std::vector <Matrix <T>>> grads(size);
			if (threads == 1) {
				std::cout << " [";
				
				for (int i = 0; i < size; i++) {
					Vector <T> actual = compute(ins[i]);

					if (__cmp(actual, outs[i]))
						passed++;

					grads[i] = gradient(adjusted(0.7), ins[i], outs[i], __cost);
					
					opt_error += (*__cost)(outs[i], actual)[0];
					per_error += 100 * (actual - outs[i]).norm()/outs[i].norm();

					if (printing) {
						int delta = (len * (i + 1))/size;
						for (int i = 0; i < delta - bars; i++) {
							std::cout << "=";
							std::cout.flush();
						}

						bars = delta;
					}
				}

				std::cout << "]";
			} else {
				std::vector <std::thread> army;
				
				double *optes = new double[threads];
				double *peres = new double[threads];
				double *pass = new double[threads];

				auto proc = [&](size_t offset) {
					std::vector <Vector <T>> aloc;
					std::vector <Vector <T>> zloc;

					for (int i = offset; i < size; i += threads) {
						Vector <T> actual = compute(ins[i], aloc, zloc);

						// TODO: Determine whether the the
						// following if statement is
						// hazardous.
						if (__cmp(actual, outs[i]))
							pass[offset]++;

						grads[i] = gradient(adjusted(0.7), aloc, zloc, ins[i], outs[i], __cost);
						
						optes[offset] += (*__cost)(outs[i], actual)[0];
						peres[offset] += 100 * (actual - outs[i]).norm()/outs[i].norm();
					}
				};

				for (int i = 0; i < threads; i++) {
					optes[i] = peres[i] = 0;

					army.push_back(std::thread(proc, i));
				}

				for (int i = 0; i < threads; i++) {
					opt_error += optes[i];
					per_error += peres[i];
					passed += pass[i];

					army[i].join();
				}

				// Free resources
				delete[] optes;
				delete[] peres;
				delete[] pass;
			}

			end = clk.now();

			std::vector <Matrix <T>> grad = grads[0];
			for (size_t i = 1; i < grads.size(); i++) {
				for (size_t j = 0; j < grad.size(); j++)
					grad[j] += grads[i][j];
			}
				
			for (size_t j = 0; j < grad.size(); j++)
				grad[j] /= (double) size;

			apply_gradient(grad, alpha, 0.7);
			
			total = clk.now();

			double avg_time = std::chrono::duration_cast
				<std::chrono::microseconds> (end - start).count();
			double tot_time = std::chrono::duration_cast
				<std::chrono::microseconds> (total - start).count();
			avg_time /= size;

			if (printing) {
				std::cout << " passed: " << passed << "/" << size << " = "
					<< std::fixed << std::showpoint << std::setprecision(2)
					<< 100 * ((double) passed)/size << "%, "
					<< "µ-err: "
					<< std::setw(width) << std::fixed
					<< std::showpoint << std::setprecision(2)
					<< per_error/size << "%, "
					<< "µ-time: " << avg_time << " µs"
					<< std::endl;
			}
			
			return {passed, opt_error, tot_time};
		}

		template <class T>
		template <size_t threads>
		typename NeuralNetwork <T> ::TrainingStatistics NeuralNetwork <T>
			::epochs(const DataSet <T> &ins,
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
				for (int i = 0; i < ins_batched.size(); i++) {
					TrainingStatistics result = train <threads> (ins_batched[i],
						outs_batched[i], lr, i + 1, printing);

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
