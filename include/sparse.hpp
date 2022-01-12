#ifndef SPARSE_H_
#define SPARSE_H_

// C++ headers
#include <cstdlib>

// Engine headers
#include "matrix.hpp"

namespace zhetapi {

template <class T>
class SparseMatrix {
	struct elem {
		elem *	_next = nullptr;	// Next element
		size_t	_ci = 0;		// Column index
	};

	elem **	_rows = nullptr;
public:
	SparseMatrix();
	SparseMatrix(const Matrix <T> &, T);
};

template <class T>
SparseMatrix <T> ::SparseMatrix() {}

template <class T>
SparseMatrix <T> ::SparseMatrix(const Matrix <T> &mat, T exc)
{
	size_t rs = mat.get_rows();
	size_t cs = mat.get_cols();

	_rows = new elem[rs];
	for (size_t i = 0; i < rs; i++) {
		_rows[i] = new elem;

		for (size_t i = 0; i < cs; i++) {

		}
	}
}

}

#endif
