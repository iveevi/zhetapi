#ifndef FUNC_STACK_H_
#define FUNC_STACK_H_

// Custom Built Libraries
#include "stack.h"
#include "functor.h"

template <class T>
class func_stack : public splay_stack <functor <T>> {
public:
	using nfe = not_found_exception;
	using node = typename splay_stack <variable <T>> ::node;

	functor <T> &get(const std::string &);
	const functor <T> &find(const std::string &);
protected:
	virtual void splay(node *(&), const std::string &);
};

template <class T>
functor <T> &func_stack <T> ::get(const std::string &id)
{
	splay(this->m_root, id);

	if (this->m_root->val.symbol() != id)
		throw nfe();
	
	return this->m_root->val;
}

template <class T>
const functor <T> &func_stack <T> ::find(const std::string &id)
{
	splay(this->m_root, id);

	if (this->m_root->val.symbol() != id)
		throw nfe();
	
	return this->m_root->val;
}

// Note: Varialbe comparison has been
// moved to name comparison instead
// of value comparison

template <class T>
void func_stack <T> ::splay(node *(&nd), const std::string &id)
{
	node *rt = nullptr;
	node *lt = nullptr;
	node *rtm = nullptr;
	node *ltm = nullptr;
	
	while (nd != nullptr) {
		if (id < nd->val.symbol()) {
			if (nd->left == nullptr)
				break;
			
			if (id < nd->left->val.symbol()) {
				this->rotate_left(nd);

				if (nd->left == nullptr)
					break;
			}

			if (rt == nullptr)
				rt = new node {variable <T> (), nullptr, nullptr};
		

			if (rtm == nullptr) {
				rt->left = nd;
				rtm = rt;
			} else {
				rtm->left = nd;
			}

			rtm = rtm->left;
			
			nd = rtm->left;
			rtm->left = nullptr;
		} else if (id > nd->val.symbol()) {
			if (nd->right == nullptr)
				break;
			
			if (id > nd->right->val.symbol()) {
				this->rotate_right(nd);
				if (nd->right == nullptr)
					break;
			}

			if (lt == nullptr)
				lt = new node {variable <T> (), nullptr, nullptr};

			if (ltm == nullptr) {
				lt->right = nd;
				ltm = lt;
			} else {
				ltm->right = nd;
			}

			ltm = ltm->right;

			nd = ltm->right;
			ltm->right = nullptr;
		} else {
			break;
		}
	}
	
	if (lt != nullptr) {
		ltm->right = nd->left;
		nd->left = lt->right;
	}

	if (rt != nullptr) {
		rtm->left = nd->right;
		nd->right = rt->left;
	}
}

#endif
