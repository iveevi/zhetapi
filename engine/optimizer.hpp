#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

namespace zhetapi {

namespace ml {

template <class T>
class Optimizer {
public:
	virtual void apply_gradient(Matrix <T> *) = 0;
};

}

}

#endif
