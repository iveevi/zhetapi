#ifndef UTILITY_H_
#define UTILITY_H_

#include <vector>
#include <utility>

#include "Matrix.h"
#include "Vector.h"

namespace utility {

	template <class T>
	std::vector <Vector <T>> gram_shmidt(const std::vector <Vector <T>> &span) {
		assert(span.size());

		std::vector <Vector <T>> basis = {span[0]};
		
		Vector <T> nelem;
		for (size_t i = 1; i < span.size(); i++) {
			nelem = span[i];

			for (size_t j = 0; j < i; j++) {
				nelem = nelem - (inner(span[i], basis[j])
						/ inner(basis[j], basis[j]))
						* basis[j];
			}

			basis.push_back(nelem);
		}

		return basis;
	}
	
	template <class T>
	std::vector <Vector <T>> gram_shmidt_normalized(const std::vector <Vector <T>> &span) {
		assert(span.size());

		std::vector <Vector <T>> basis = {span[0].normalize()};
	
		Vector <T> nelem;
		for (size_t i = 1; i < span.size(); i++) {
			nelem = span[i];

			for (size_t j = 0; j < i; j++) {
				nelem = nelem - (inner(span[i], basis[j])
						/ inner(basis[j], basis[j]))
						* basis[j];
			}

			basis.push_back(nelem.normalize());
		}

		return basis;
	}

	template <class T>
	const Function <T> &lagrange_interpolate(const std::vector <std::pair <T, T>> &points)
	{
		Function <T> *out = new Function <T> ("f(x) = 0");

		for (int i = 0; i < points.size(); i++) {
			Function <T> *term = new Function <T> ("f(x) = " + to_string(points[i].second));

			for (int j = 0; j < points.size(); j++) {
				if (i == j)
					continue;

				Function <T> *tmp = new Function <T> ("f(x) = (x - " + to_string(points[j].first)
					+ ")/(" + to_string(points[i].first) + " - " + to_string(points[j].first) + ")");

				*term = (*term * *tmp);
			}

			*out = (*out + *term);
		}

		return *out;
	}

	/* Make member */
	template <class T>
	std::pair <Matrix <T> , Matrix <T>> lu_factorize(const Matrix <T> &a)
	{
		assert(a.get_rows() == a.get_cols());

		size_t size = a.get_rows();
		
		Matrix <T> u(size, size, 0);
		Matrix <T> l(size, size, 0);

		T value;
		for (size_t i = 0; i < size; i++) {
			for (int j = i; j < size; j++) {
				value = 0;

				for (int k = 0; k < i; k++)
					value += l[i][k] * u[k][j];

				u[i][j] = a[i][j] - value;
			}

			for (int j = i; j < size; j++) {
				value = 0;

				if (i == j) {
					l[i][i] = 1;
				} else {
					value = 0;

					for (int k = 0; k < i; k++)
						value += l[j][k] * u[k][i];

					l[j][i] = (a[j][i] - value) / u[i][i];
				}
			}
		}

		return {l, u};
	}

	template <class T>
	const Vector <T> &solve_linear_equation(const Matrix <T> &a, const Vector <T> &b)
	{
		std::pair <Matrix <T>, Matrix <T>> out = lu_factorize(a);

		Matrix <T> L = out.first;
		Matrix <T> U = out.second;

		size_t size = a.get_rows();

		Vector <T> *y = new Vector <T> (size, -1);
		Vector <T> *x = new Vector <T> (size, -1);

		T value;
		for (size_t i = 0; i < size; i++) {
			value = 0;

			for (size_t j = 0; j < i; j++)
				value += (*y)[j] * L[i][j];

			(*y)[i] = (b[i] - value)/L[i][i];
		}

		for (size_t i = size - 1; i != -1; i--) {
			value = 0;

			for (size_t j = size - 1; j > i; j--)
				value += (*x)[j] * U[i][j];

			(*x)[i] = ((*y)[i] - value)/U[i][i];
		}

		return *x;
	}

	template <class T>
	const Function <T> &reduced_polynomial_fitting(const std::vector <std::pair <T, T>> &points)
	{
		size_t degree = points.size();

		Matrix <T> coefficients(degree, degree, [&](size_t i, size_t j) {
			return pow(points[i].first, degree - (j + 1));
		});

		Vector <T> out(degree, [&](size_t i) {
			return points[i].second;
		});
		
		Vector <T> constants = solve_linear_equation(coefficients, out);
		
		std::string str = "f(x) = 0";

		for (size_t i = 0; i < degree; i++) {
			if (constants[i] == 0)
				continue;

			string sign = " + ";
			if (constants[i] < 0)
				sign = " - ";

			str += sign + std::to_string(abs(constants[i])) + "x^" + std::to_string(degree - (i + 1));
		}

		Function <T> *ftr = new Function <T> (str);

		return *ftr;
	}

	template <class T>
	std::pair <T, std::vector <T>> gradient_descent(std::vector <pair <T, T>> data,
		std::vector <T> weights, Function <T> ftr, size_t in,
		size_t reps, size_t rounds, T _gamma, T diff, T eps)
	{
		table <T> tbl {ftr};

		config <T> *cptr = new config <T> {};

		std::vector <variable <T>> pars;
		std::vector <variable <T>> vars;

		std::vector <node <T> *> lvs;

		for (size_t i = 0; i < ftr.ins(); i++) {
			pars.push_back(ftr[i]);

			if (i != in)
				vars.push_back(ftr[i]);

			lvs.push_back(new node <T> {new variable <T> 
					{ftr[i].symbol(), true}, {}, cptr});
		}

		pars.push_back(variable <T> {"y", true});

		node <T> *pk = new node <T> {cptr->alloc_opn(op_exp), {
			new node <T> {cptr->alloc_opn(op_sub), {
				new node <T> {new variable <T> {"y", true}, {}, cptr},
				new node <T> {new Function <T> {ftr}, lvs, cptr}
			}, cptr},
			new node <T> {new operand <T> (2), {}, cptr},
		}, cptr};

		pk->reparametrize(pars);

		Function <T> cost {"cost", pars, pk};

		std::vector <Function <T>> gradients;

		for (size_t i = 0; i < ftr.ins(); i++) {
			if (i == in)
				continue;

			gradients.push_back(cost.differentiate(ftr[i].symbol()));
		}

		T old;
		T value;

		T err;

		T x;
		T y;

		T gamma;
		T best;
		
		best = 0;
		for (auto pnt : data) {
			std::vector <T> ins = weights;

			ins.push_back(pnt.first);
			ins.push_back(pnt.second);

			best += cost.compute(ins);
		}

		std::vector <T> bvls = weights;
		for (int n = 0; n < reps; n++) {
			std::vector <T> ws;

			for (auto vl : weights)
				ws.push_back(vl + diff * n);

			gamma = _gamma;

			std::vector <T> pvls;

			for (auto vl : ws)
				pvls.push_back(vl);

			old = 0;
			for (auto pnt : data) {
				std::vector <T> ins = pvls;

				ins.push_back(pnt.first);
				ins.push_back(pnt.second);

				old += cost.compute(ins);
			}

			std::vector <T> wds(ws.size(), 0.0);

			std::vector <T> cvls;
			for (int i = 0; i < rounds; i++) {
				cvls.clear();

				for (auto vl : ws)
					cvls.push_back(vl);

				for (auto pnt : data) {
					x = pnt.first;
					y = pnt.second;

					std::vector <T> ivls = cvls;

					ivls.push_back(x);
					ivls.push_back(y);
					
					err = cost.compute(ivls);

					for (size_t i = 0; i < wds.size(); i++)
						wds[i] += gamma * gradients[i].compute(ivls);
				}

				for (size_t i = 0; i < ws.size(); i++)
					ws[i] -= wds[i]/data.size();

				value = 0;
				for (auto pnt : data) {
					std::vector <T> ins = ws;

					ins.push_back(pnt.first);
					ins.push_back(pnt.second);

					value += cost.compute(ins);
				}

				if (old <= value) {
					for (size_t i = 0; i < ws.size(); i++)
						ws[i] = cvls[i];

					gamma *= 2;
				} else {
					gamma /= 2;

					if (old - value < eps)
						break;

					old = value;
				}

				gamma = min(100.0, gamma);
			}

			if (best > old) {
				diff *= -0.5;

				bvls = ws;
			} else {
				diff *= -2;
			}
			
			best = min(best, old);
		}
		
		return {best, bvls};
	}

	// exception for when root finding
	// hits an extrema. Later store the
	// vector wchih produced this exception.
	class extrema_exception {};

	template <class T>
	Vector <T> find_root(Function <T> ftr, const Vector <T> &guess, size_t rounds)
	{
		Vector <T> sol = guess;

		Vector <Function <T>> J_raw(ftr.ins(), [&](size_t i) {
			// allow differentiation in index w/ respect
			// to the parameters/variables
			return new Function <T> (ftr.differentiate(ftr[i].symbol()));
		});

		for (size_t i = 0; i < rounds; i++) {
			T val = ftr(sol);

			if (val == 0)
				break;

			Vector <T> J_mod(ftr.ins(), [&](size_t i) {
				T tmp = J_raw[i](sol);

				if (tmp == 0)
					throw extrema_exception();

				return val/tmp;
			});

			sol -= J_mod;
		}

		return sol;
	}
	
};

#endif
