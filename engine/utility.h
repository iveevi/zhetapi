#ifndef UTILITY_H_
#define UTILITY_H_

#include "element.h"

namespace utility {
	template <class T>
	std::vector <element <T>> gram_shmidt(const std::vector
			<element <T>> &span) {
		assert(span.size());

		std::vector <element <T>> basis = {span[0]};
		
		element <double> nelem;
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
	std::vector <element <T>> gram_shmidt_normalized(const std::vector
			<element <T>> &span) {
		assert(span.size());

		std::vector <element <T>> basis = {span[0].normalize()};
	
		element <double> nelem;
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
	const functor <T> &interpolate_lagrange(const std::vector <std::pair
			<T, T>> &points)
	{
		functor <T> *out = new functor <T> ("f(x) = 0");

		for (int i = 0; i < points.size(); i++) {
			functor <T> *term = new functor <T> ("f(x) = " + to_string(points[i].second));

			for (int j = 0; j < points.size(); j++) {
				if (i == j)
					continue;

				functor <T> *tmp = new functor <T> ("f(x) = (x - " + to_string(points[j].first)
					+ ")/(" + to_string(points[i].first) + " - " + to_string(points[j].first) + ")");

				*term = (*term * *tmp);
			}

			*out = (*out + *term);
		}

		return *out;
	}

	template <class T>
	std::pair <matrix <T> , matrix <T>> lu_factorize(const matrix <T> &a)
	{
		assert(a.get_rows() == a.get_cols());

		size_t size = a.get_rows();
		
		matrix <T> u(size, size, 0);
		matrix <T> l(size, size, 0);

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
};

#endif
