#ifndef GNN_H_
#define GNN_H_

// C/C++ headers
#include <queue>
#include <set>
#include <vector>

// Engine headers
#include <netnode.hpp>

namespace zhetapi {

namespace ml {

/*
 * General neural network (GNN):
 *
 * Represents a neural network whose structure is theoretically unlimited, i.e.
 * a neural network with various kinds of connections (skip connections, etc.)
 * and different types of layers (deep layer, convolutional layer, recurrent layer).
 */
template <class T = double>
class GNN {
	std::vector <NetNode <T> *>	__ins	= {};
	std::vector <NetNode <T> *>	__outs	= {};

	// Variadic constructor helpers
	void init(NetNode <T> *);

	template <class ... U>
	void init(NetNode <T> *, U ...);

	// Initialize outs
	void getouts();
public:
	GNN();
	explicit GNN(NetNode <T> *);
	explicit GNN(const std::vector <NetNode <T> *> &);

	// Variadic constructor
	template <class ... U>
	explicit GNN(U ...);

	// Extraction
	NetNode <T> &operator[](size_t);

	const NetNode <T> &operator[](size_t) const;

	// Passing
	void pass(std::vector <Tensor <T>> &) const;
	void pass(std::vector <Tensor <T>> &&) const;

	void trace() const;
};

template <class T>
GNN <T> ::GNN() {}

template <class T>
GNN <T> ::GNN(NetNode <T> *nnptr) : __ins({nnptr})
{
	getouts();
}

template <class T>
GNN <T> ::GNN(const std::vector <NetNode <T> *> &ins)
		: __ins(ins) {}

template <class T>
template <class ... U>
GNN <T> ::GNN(U ... args)
{
	init(args...);
}

template <class T>
void GNN <T> ::init(NetNode <T> *nnptr)
{
	__ins.push_back(nnptr);

	getouts();
}

template <class T>
template <class ... U>
void GNN <T> ::init(NetNode <T> *nnptr, U ... args)
{
	__ins.push_back(nnptr);

	init(args...);
}

template <class T>
void GNN <T> ::getouts()
{
	// Set of visited nodes
	std::set <NetNode <T> *> vis;

	// BFS queue
	std::queue <NetNode <T> *> queue;

	for (NetNode <T> *nnptr : __ins)
		queue.emplace(nnptr);
	
	while (!queue.empty()) {
		NetNode <T> *cptr = queue.top();

		queue.pop();

		if (vis.find() != vis.end())
			continue;

		auto vfrw = cptr->forward();
		if (vfrw.empty()) {
			__outs.push_back(cptr);
		} else {
			for (auto frw : vfrw)
				queue.push(frw->__fr);
		}
	}
}

/**
 * @brief This specific overload (used for modifying the state of an input)
 * can be used to alter the properties of a specific input node.
 */
template <class T>
NetNode <T> &GNN <T> ::operator[](size_t i)
{
	return *(__ins[i]);
}

/**
 * @brief This specific overload is used to extract the output nodes, from
 * which the Tensors in the pipes can be accessed.
 */
template <class T>
const NetNode <T> &GNN <T> ::operator[](size_t i) const
{
	return *(__outs[i]);
}

// Passing
template <class T>
void GNN <T> ::pass(std::vector <Tensor <T>> &args) const
{
	size_t i = 0;
	while (!args.empty() && i < __ins.size()) {
		__ins[i].pass(args);

		i++;
	}
}

template <class T>
void GNN <T> ::pass(std::vector <Tensor <T>> &&rargs) const
{
	std::vector <Tensor <T>> args = std::move(rargs);

	pass(args);
}

template <class T>
void GNN <T> ::trace() const
{
	for (NetNode <T> *nn : __ins)
		nn->trace();
}

}

}

#endif
