#ifndef ERF_DERIVATIVES_H_
#define ERF_DERIVATIVES_H_

namespace zhetapi {

namespace ml {

// Squared error
template <class T>
class _DSE : public Erf <T> {
public:
	__cuda_dual__
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		return -T(2) * (comp - in);
	}
};

// M squared error
template <class T>
class _DMSE : public Erf <T> {
public:
	__cuda_dual__
	Vector <T> operator()(const Vector <T> &comp, const Vector <T> &in) const {
		return -T(2)/T(comp.size()) * (comp - in);
	}
};

}

}

#endif
