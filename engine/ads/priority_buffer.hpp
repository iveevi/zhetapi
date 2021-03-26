#ifndef PRIORITY_BUFFER_H_
#define PRIORITY_BUFFER_H_

// C++ headers
#include <cstdlib>
#include <functional>

namespace zhetapi {

namespace ads {

/*
 * The PriorityBuffer class should be viewed as a circular list, where elements
 * are inserted and removed at each iteration of some algorithm. It is also a
 * threaded red-black tree; searching for the smallest and largest elements are
 * fast.
 *
 * @tparam T The type of data stored.
 * @tparam C The comparator function.
 */
template <class T, class C = std::less <T>>
class PriorityBuffer {
	struct node {
		T	_val;

		node *	_tnext;	// Next value in time
		node *	_tprev;	// Previous value in time

		node *	_snext;	// Next value in sorting
		node *	_sprev;	// Previous value in sorting

		node *	_left;
		node *	_right;
	};

	node *	_head = nullptr;
	node *	_last = nullptr;

	size_t	_max_size = 0;

	C	_cmp = C();		// Comparison function
public:
	PriorityBuffer();
	PriorityBuffer(size_t);

	void insert(const T &);
	
	void resize(size_t);
};

// Constructors
template <class T, class C>
PriorityBuffer <T, C> ::PriorityBuffer() {}

template <class T, class C>
PriorityBuffer <T, C> ::PriorityBuffer(size_t max_size)
		: _max_size(max_size) {}

template <class T, class C>
void PriorityBuffer <T, C> ::resize(size_t new_size)
{
	_max_size = new_size;
}

}

}

#endif
