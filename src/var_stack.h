#ifndef VAR_STACK_H_
#define VAR_STACK_H_

// Custom Built Libraries
#include "stack.h"
#include "variable.h"

template <class T>
class var_stack : public splay_stack <variable <T>> {
public:
	using nfe = typename splay_stack <variable <T>> ::not_found_exception;
	using node = typename splay_stack <variable <T>> ::node;

	const variable <T> &find(const std::string &);
protected:
	virtual void splay(node *(&), const std::string &);
private:
	node *(&rptr) = this->m_root;
};

template <class T>
const variable <T> &var_stack <T> ::find(const std::string &id)
{
	splay(rptr, id);

	if (rptr->val.symbol() != id)
		throw nfe();
	
	return rptr->val;
}

#endif