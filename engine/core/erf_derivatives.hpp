#ifndef ERF_DERIVATIVES_H_
#define ERF_DERIVATIVES_H_

namespace zhetapi {

namespace ml {

// Squared error
template <class T>
class __DSquaredError : public Erf <T> {
public:
	__cuda_dual_prefix
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		return -T(2) * (comp - in);
	}
};

// Mean squared error
template <class T>
class __DMeanSquaredError : public Erf <T> {
public:
	__cuda_dual_prefix
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		return -T(2)/T(comp.size()) * (comp - in);
	}
};

}

}

#endif