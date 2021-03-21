#ifndef STD_FILTERS_H_
#define STD_FILTERS_H_

// Engine headrers
#include <filter.hpp>
#include <matrix.hpp>
#include <vector.hpp>
#include <image.hpp>

#include <std/initializers.hpp>

namespace zhetapi {

namespace ml {

template <class T = double>
class FeedForward : public Filter <T> {
	Matrix <T>	__weight	= Matrix <T> ();

	Activation <T> *__act		= nullptr;
	Activation <T> *__dact		= nullptr;

	long double	__dropout	= 0;

	Vector <T>	__acache	= Vector <T> ();
	Vector <T>	__zcache	= Vector <T> ();

	// For batch inputs
	// Matrix <T>	__Acache;
	// Matrix <T>	__Zcache;
public:
	// Input size, output size
	FeedForward(size_t isize, size_t osize, Activation <T> *act, std::function <T ()> init = RandomInitializer <T> ())
			: __weight(isize, osize + 1),	// +1 for bias
			__act(act->copy()),
			__dact(act->derivative())
	{
		__weight.randomize(init);
	}

	void propogate(const Pipe <T> &in, Pipe <T> &out)
	{
		// Slice the input (+1 for bias)
		Vector <T> vin = (in[0]->cast_to_vector()).append_above(1);

		__acache = vin;

		Vector <T> mul = __weight * vin;

		__zcache = __dact->compute(mul);

		// Send to output pipe
		*out[0] = __act->compute(mul);
	}

	void gradient(const Pipe <T> &delin, Pipe <T> &grads)
	{
		// TODO: Check sizes later

		// Move shur/stable shur to tensor base
		*delin[0] = shur(delin[0]->cast_to_vector(), __zcache);

		Matrix <T> J = delin[0]->cast_to_vector() * __acache.transpose();

		// Use the kernel function here
		*grads[0] = J;
	}

	void apply_gradient(const Pipe <T> &grads)
	{
		// TODO: check with gradeint checking
		Matrix <T> J = grads[0]->cast_to_matrix(
				__weight.get_rows(),
				__weight.get_cols());

		__weight += J;
	}
};

#define for_img(i, j, w, h)		\
	for (int i = 0; i < w; i++) {	\
	for (int j = 0; j < h; j++)

// Assumes that the input tensor is an image
template <class T>
class Convolution : public Filter <T> {
	Matrix <T>	__filter;
	size_t		__dim;
	
	// Type aliases
	using byte = image::byte;
	using mbyte = Matrix <byte>;
	using vbyte = Vector <byte>;
	using vfilt = Vector <T>;
public:
	Convolution(const Matrix <T> &filter)
			: __filter(filter), 
			__dim(filter.get_rows()) {}

	// Assume equal padding for now
	image::Image process(const image::Image &in, int depth = -1) {
		image::Image out = in;

		int w = in.width();
		int h = in.height();
		int c = in.channels();

		// depth = c;
		// Choose color channels only
		if (depth < 0)
			depth = (c > 1) ? c - 1 : c;

		int n = (__dim - 1)/2;

		byte *data = in.__array;

		using namespace std;
		for_img(x, y, w, h) {
			vbyte t(depth, byte(0));
			
			int ymin = y - n;
			int ymax = y + n;

			int xmin = x - n;
			int xmax = x + n;

			Vector <T> tmp(depth, T(0));
			for (int k = 0; k < __dim; k++) {
				size_t ti = x + k - n;

				if (xmin + k < 0 || xmin + k >= h)
					continue;

				size_t off = ymin;
				size_t len = __dim;

				if (ymin < 0) {
					off = 0;
					len += ymin;
				}

				if (ymax >= w)
					len -= (ymax - w + 1);

				size_t i = c * ((x + k - n) * w + off);

				byte *img = &(data[i]);
				T *flt = &(__filter[k][off - ymin]);
				
				for (size_t ch = 0; ch < depth; ch++) {
					T s = 0;

					for (size_t i = 0; i < len; i++)
						s += flt[i] * ((T) img[i * c + ch]);

					tmp[ch] += s;
				}
			}
			
			for (size_t i = 0; i < depth; i++)
				t[i] = (tmp[i] > 0) ? tmp[i] : 0;

			out.set({x, y}, t);
		}}

		return out;
	}
};

}

}

#endif
