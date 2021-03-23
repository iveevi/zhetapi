#ifndef SPARSE_H_
#define SPARSE_H_

// C++ headers
#include <cstdlib>

// Engine headers
#include <matrix.hpp>

namespace zhetapi {

template <class T>
class SparseMatrix {
	struct elem {
		elem *	__next = nullptr;	// Next element
		size_t	__ci = 0;		// Column index
	};

	elem **	__rows = nullptr;
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

	__rows = new elem[rs];
	for (size_t i = 0; i < rs; i++) {
		__rows[i] = new elem;

		for (size_t i = 0; i < cs; i++) {

		}
	}
}

}

#endif
