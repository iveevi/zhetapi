#ifndef VECTOR_CUH_
#define VECTOR_CUH_

namespace zhetapi {

// hc is half copy vector
template <class T>
void Vector <T> ::cuda_read(Vector <T> *hc)
{
	if (hc->_size != this->_size) {
		// Add a clear array function (and clear dim)
		delete[] this->_array;

		this->_array = new T[hc->_size];
	}

	hc->_arena->read(this->_array, hc->_array, hc->_size);
	hc->_arena->read(this->_dim, hc->_dim, 2);
}

// returns a vector with only _dim and _array in device memory
// requires the callee to be fully in host memory
template <class T>
Vector <T> *Vector <T> ::cuda_half_copy(NVArena *arena) const
{
	size_t *dim = arena->alloc <size_t> (2);
	T *array = arena->alloc <T> (this->_size);

	arena->write(dim, this->_dim, 2);
	arena->write(array, this->_array, this->_size);

	// Host copy
	Vector <T> *hc = new Vector <T>;
	memcpy(hc, this, sizeof(Vector <T>));

	// Edit hc with the correct values
	hc->_array = array;
	hc->_dim = dim;
	hc->_on_device = true;
	hc->_arena = arena;

	return hc;
}

// returns a vector fully in device memory
// requires the callee to be partially in device memory (_dim and _array)
template <class T>
Vector <T> *Vector <T> ::cuda_full_copy(NVArena *arena)
{
	Vector <T> *fc = arena->alloc <Vector <T>> ();
	arena->write(fc, this);
	// cudaMemcpy(fc, this, sizeof(Vector <T>), cudaMemcpyHostToDevice);
	return fc;
}

}

#endif
