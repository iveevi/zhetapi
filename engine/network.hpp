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
			std::function <T ()>		__random;
			std::size_t			__isize;
			std::size_t			__osize;

			std::vector <Vector <T>>	__xs;
			std::vector <Vector <T>>	__dxs;
		public:
			NeuralNetwork(const std::vector <Layer> &, const std::function <T ()> &);

			Vector <T> compute(const Vector <T> &);
			Vector <T> operator()(const Vector <T> &);

			std::vector <Matrix <T>> gradient(const Vector <T> &,
					const Vector <T> &, Optimizer <T> *);

			void learn(const Vector <T> &, const Vector <T> &,
					Optimizer <T> *, T);

			void train(Optimizer <T> *, const std::vector <Vector<T>> &,
					const std::vector <Vector <T>> &,
					const std::function <bool (const Vector <T> , const Vector <T>)> &,
					bool = false);
			void epochs(size_t, Optimizer <T> *, const std::vector <Vector <T>> &,
					const std::vector <Vector <T>> &,
					const std::function <bool (const Vector <T> , const Vector <T>)> &,
					bool = false);

			void randomize();

			// Printing weights
			void print() const;
		};

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
			}
		}

		template <class T>
		Vector <T> NeuralNetwork <T> ::compute(const Vector <T> &in)
		{
			assert(in.size() == __isize);

			Vector <T> prv = in;
			Vector <T> tmp = (*__layers[0].second)(prv);

			__xs.clear();
			__dxs.clear();

			for (size_t i = 0; i < __weights.size(); i++) {
				__xs.insert(__xs.begin(), tmp.append_above(T (1)));

				prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));
				tmp = (*__layers[i + 1].second)(prv);
				
				__dxs.insert(__dxs.begin(), (*__layers[i].second->derivative())(prv));
			}
			
			return tmp;
		}

		template <class T>
		Vector <T> NeuralNetwork <T> ::operator()(const Vector <T> &in)
		{
			assert(in.size() == __isize);

			Vector <T> prv = in;
			Vector <T> tmp = (*__layers[0].second)(prv);

			__xs.clear();
			__dxs.clear();

			for (size_t i = 0; i < __weights.size(); i++) {
				__xs.insert(__xs.begin(), tmp.append_above(T (1)));

				prv = __weights[i] * Matrix <T> (tmp.append_above(T (1)));
				tmp = (*__layers[i + 1].second)(prv);
				
				__dxs.insert(__dxs.begin(), (*__layers[i].second->derivative())(prv));
			}
			
			return tmp;
		}

		template <class T>
		void print_dims(const Matrix <T> &a)
		{
			using namespace std;
			cout << "\trows: " << a.get_rows() << endl;
			cout << "\tcols: " << a.get_cols() << endl;
		}
		
		template <class T>
		std::Vector <Matrix <T>> NeuralNetwork <T> ::gradient(const Vector <T> &in,
				const Vector <T> &out, Optimizer <T> *opt)
		{
			assert(in.size() == __isize);
			assert(out.size() == __osize);
			
			Vector <T> actual = (*this)(in);

			Vector <T> delta = (*(opt->derivative()))(out, actual);

			std::vector <Matrix <T>> changes;

			int n = __weights.size();
			for (int i = n - 1; i >= 0; i--) {
				if (i != n - 1) {
					delta = __weights[i + 1].transpose() * delta;

					delta = delta.remove_top();
				}

				delta = shur(delta, __dxs[(n - 1) - i]);

				Matrix <T> xt = __xs[n - i - 1].transpose();

				changes.push_back(delta * xt);
			}

			std::reverse(changes.begin(), changes.end());

			return changes;
		}

		template <class T>
		void NeuralNetwork <T> ::learn(const Vector <T> &in, const Vector <T> &out, Optimizer <T> *opt, T alpha)
		{
			/* using namespace std;

			cout << "================================" << endl;
			
			cout << "Xs:" << endl;
			for (auto elem : __xs)
				cout << "X:\t" << elem << endl;
			
			cout << "Dxs:" << endl;
			for (auto elem : __dxs)
				cout << "Dx:\t" << elem << endl; */
			
			assert(in.size() == __isize);
			assert(out.size() == __osize);
			
			Vector <T> actual = (*this)(in);

			Vector <T> delta = (*(opt->derivative()))(out, actual);

			std::vector <Matrix <T>> changes;

			int n = __weights.size();
			for (int i = n - 1; i >= 0; i--) {
				/* cout << "=========================" << endl;

				cout << "weight[" << i << "]" << endl;
				print_dims(__weights[i]); */

				if (i != n - 1) {
					/* cout << "weight^T:" << endl;
					print_dims(__weights[i + 1].transpose()); */

					delta = __weights[i + 1].transpose() * delta;

					delta = delta.remove_top();

					/* cout << "delta: " << endl;
					print_dims(delta);

					Vector <T> tmp(__weights[i + 1].transpose() * delta);

					cout << "tmp alias:" << endl;
					print_dims(__weights[i + 1].transpose() * delta);
					
					cout << "pre tmp: " << endl;
					print_dims(tmp);

					tmp = tmp.remove_top();

					cout << "post tmp: " << endl;
					print_dims(tmp); */
					// delta = __weights[i + 1].transpose() * delta;
				}

				/* cout << "delta: " << endl;
				print_dims(delta);

				cout << "__dxs[(n - 1) - i]: " << endl;
				print_dims(__dxs[(n - 1) - i]); */

				delta = shur(delta, __dxs[(n - 1) - i]);

				Matrix <T> xt = __xs[n - i - 1].transpose();

				/* cout << "xt: " << endl;
				print_dims(xt);

				cout << "delta * xt:" << endl;
				print_dims(delta * xt); */

				changes.push_back(delta * xt);
			}

			// cout << "===P2: APPLY===" << endl;
			for (size_t i = 0; i < n; i++) {
				/* cout << "weights: " << endl;
				print_dims(__weights[n - (i + 1)]);

				cout << "changes: " << endl;
				print_dims(changes[i]); */

				__weights[n - (i + 1)] -= alpha * changes[i];
			}
		}

		template <class T>
		void NeuralNetwork <T> ::train(Optimizer <T> *opt, const std::vector<Vector<T>> &ins, const std::vector<Vector<T>> &outs, const std::function <bool (const Vector<T>, const Vector<T>)> &crit, bool print)
		{
			using namespace std;

			cout.flush();

			int passed = 0;

			double opt_error = 0;
			double per_error = 0;

			int size = ins.size();
			for (int i = 0; i < size; i++) {
				Vector <T> actual = compute(ins[i]);

				if (crit(actual, outs[i]))
					passed++;
				
				opt_error += (*opt)(outs[i], actual)[0];
				per_error += 100 * (actual - outs[i]).norm()/outs[i].norm();
			}

			if (print) {
				cout << "Summary: " << endl;
				
				cout << "\tCase passed:\t\t\t" << passed << "/" << size << endl;
				cout << "\tAverage (optimizer) error:\t" << opt_error/size << endl;
				cout << "\tAverage (percent) error:\t" << per_error/size << "%" << endl;
			}
		}

		template <class T>
		void NeuralNetwork <T> ::epochs(size_t runs, Optimizer<T> *opt, const std::vector<Vector<T>> &ins, const std::vector<Vector<T>> &outs, const std::function<bool (const Vector<T>, const Vector<T>)> &crit, bool print)
		{
			for (int i = 0; i < runs; i++) {
				if (print) {
					::std::cout << ::std::string(20, '-') << ::std::endl;
					::std::cout << "Epoch #" << (i + 1) << "\n" << ::std::endl;
				}

				train(opt, ins, outs, crit, print);
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
