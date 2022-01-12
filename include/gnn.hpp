#ifndef GNN_H_
#define GNN_H_

// C/C++ headers
#include <queue>
#include <set>
#include <vector>

// Engine headers
#include "netnode.hpp"

namespace zhetapi {

namespace ml {

/**
 * @brief General neural network (GNN):
 *
 * Represents a neural network whose structure is theoretically unlimited, i.e.
 * a neural network with various kinds of connections (skip connections, etc.)
 * and different types of layers (deep layer, convolutional layer, recurrent layer).
 * 
 * The true representation of the network is as a series of "pipes" between
 * nodes. Each of these pipes contains a Tensor object representing the pipes
 * current state of execution. The two most important sets of pipes are the
 * input and output pipes which, as their name implies, carry the inputs and
 * outputs.
 */
template <class T = double>
class GNN {
	std::vector <NetNode <T> *>	_ins	= {};
	std::vector <NetNode <T> *>	_outs	= {};

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
	inline NetNode <T> &ipipe(size_t);
	inline NetNode <T> &operator[](size_t);

	// Retrieval
	inline const NetNode <T> &opipe(size_t) const;
	inline const NetNode <T> &operator[](size_t) const;

	// Passing
	void pass(std::vector <Tensor <T>> &) const;
	void pass(std::vector <Tensor <T>> &&) const;

	void trace() const;
};

template <class T>
GNN <T> ::GNN() {}

template <class T>
GNN <T> ::GNN(NetNode <T> *nnptr) : _ins({nnptr})
{
	getouts();
}

template <class T>
GNN <T> ::GNN(const std::vector <NetNode <T> *> &ins)
		: _ins(ins) {}

template <class T>
template <class ... U>
GNN <T> ::GNN(U ... args)
{
	init(args...);
}

template <class T>
void GNN <T> ::init(NetNode <T> *nnptr)
{
	_ins.push_back(nnptr);

	getouts();
}

template <class T>
template <class ... U>
void GNN <T> ::init(NetNode <T> *nnptr, U ... args)
{
	_ins.push_back(nnptr);

	init(args...);
}

template <class T>
void GNN <T> ::getouts()
{
	// Set of visited nodes
	std::set <NetNode <T> *> vis;

	// BFS queue
	std::queue <NetNode <T> *> queue;

	for (NetNode <T> *nnptr : _ins)
		queue.emplace(nnptr);
	
	while (!queue.empty()) {
		NetNode <T> *cptr = queue.top();

		queue.pop();

		if (vis.find() != vis.end())
			continue;

		auto vfrw = cptr->forward();
		if (vfrw.empty()) {
			_outs.push_back(cptr);
		} else {
			for (auto frw : vfrw)
				queue.push(frw->_fr);
		}
	}
}

/**
 * @brief Modifies an input pipe (when assigned, such as `gnn.ipipe() =
 * tensor`).
 * 
 * @param i the input pipe index.
 */
template <class T>
inline NetNode <T> &GNN <T> ::ipipe(size_t i)
{
	return *(_ins[i]);	
}

/**
 * @brief Modifies an input pipe (when assigned, such as `gnn[0] = tensor`).
 * 
 * @param i the input pipe index.
 */
template <class T>
inline NetNode <T> &GNN <T> ::operator[](size_t i)
{
	return *(_ins[i]);
}

/**
 * @brief Retrieves an output pipe.
 * 
 * @param i the output pipe index.
 */
template <class T>
inline const NetNode <T> &GNN <T> ::opipe(size_t i) const
{
	return *(_outs[i]);
}

/**
 * @brief Retrieves an output pipe.
 * 
 * @param i the output pipe index.
 */
template <class T>
inline const NetNode <T> &GNN <T> ::operator[](size_t i) const
{
	return *(_outs[i]);
}

// Passing
template <class T>
void GNN <T> ::pass(std::vector <Tensor <T>> &args) const
{
	size_t i = 0;
	while (!args.empty() && i < _ins.size()) {
		_ins[i].pass(args);

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
	for (NetNode <T> *nn : _ins)
		nn->trace();
}

}

}

#endif
