#ifndef STACKS_H_
#define STACKS_H_

#include <iostream>

// remove later
class not_found_exception {};
class empty_tree_exception {};

// the stack class acts
// as a storage class for
// various objects, specifically
// variable and function stacks
// It does not, however, represent
// a LIFO structure. It is instead
// a splay tree, so that the program
// can use the property of temporal
// locality - variables being used
// will likely be used in the near
// future

template <class T>
class splay_stack {
public:
	struct node {
		T val;
		node *left;
		node *right;
	};

	splay_stack();
	splay_stack(const splay_stack &);

	~splay_stack();

	// dont use this in inherited classes,
	// instead use specific keys to search
	// for variables
	const T &find(const T &) const;

	bool empty() const;
	std::size_t size() const;

	void clear();

	const splay_stack &operator= (const splay_stack &);

	bool insert(const T &);
	bool remove(const T &);
protected:
	node *clone(node *) const;

	void clear(node *(&)) const;

	void rotate_left(node *(&)) const;
	void rotate_right(node *(&)) const;

	// overload this in derived classes
	virtual void splay(node *(&), const T &) const;
private:
	std::size_t m_size;
	node *m_root;
};

/* splay_tree class constructors
 * and destructors */
template <class T>
splay_stack <T> ::splay_stack() : m_size(0), m_root(nullptr) {}

template <class T>
splay_stack <T> ::splay_stack(const splay_stack &rhs) : m_size(0),
	m_root(nullptr)
{
	*this = rhs;
}

template <class T>
splay_stack <T> ::~splay_stack()
{
	clear();
}

/* splay_tree other member
 * functions (public interface) */
template <class T>
const T &splay_stack <T> ::find(const T &key) const
{
	splay(m_root, key);

	if (m_root->val != key)
		throw not_found_exception();

	return m_root->val;
}

template <class T>
bool splay_stack <T> ::empty() const
{
	return m_size == 0;
}

template <class T>
std::size_t splay_stack <T> ::size() const
{
	return m_size;
}

template <class T>
void splay_stack <T> ::clear()
{
	clear(m_root);
}

template <class T>
const splay_stack <T> &splay_stack <T> ::operator= (const splay_stack<T> &rhs)
{
	if (rhs != *this) {
		clear();
		m_root = clone(rhs.m_root);
		m_size = rhs.m_size;
	}

	return *this;
}

template <class T>
bool splay_stack <T> ::insert(const T &val)
{
	if (m_root == nullptr) {
		m_root = new node {val, nullptr, nullptr};
		m_size++;
		return true;
	}

	splay(m_root, val);

	node *temp = new node {val, nullptr, nullptr};

	if (val < m_root->val) {
		temp->left = m_root->left;
		temp->right = m_root->right;

		temp->right->left = nullptr;
		m_root = temp;
		m_size++;

		return true;
	}

	if (val > m_root->val) {
		temp->left = m_root;
		temp->right = m_root->right;

		temp->left->right = nullptr;
		m_root = temp;
		m_size++;

		return true;
	}

	return false;
}

template <class T>
bool splay_stack <T> ::remove(const T &val)
{
	if (!m_size)
		return false;

	splay(m_root, val);

	if (val != m_root->val)
		return false;

	node *nroot;
	if (m_root->left == nullptr) {
		nroot = m_root->left;
	} else {
		nroot = m_root->left;
		splay(nroot, val);

		nroot->right = m_root->right;
	}

	delete m_root;

	m_root = nroot;
	m_size--;

	return true;
}

/* splay_tree other member
 * functions (protected interface) */
template <class T>
typename splay_stack <T> ::node *splay_stack <T>
	::clone(node *nd) const
{
	node *nnode;

	if (nd == nullptr)
		return nullptr;

	nnode = new node {nd->val, clone(nd->left),
		clone(nd->right)};

	return nnode;
}

template <class T>
void splay_stack <T> ::clear(node *(&nd)) const
{
	
}

template <class T>
void splay_stack <T> ::rotate_left(node *(&nd)) const
{

}

template <class T>
void splay_stack <T> ::rotate_right(node *(&nd)) const
{

}

template <class T>
void splay_stack <T> ::splay(node *(&nd), const T &val) const
{

}

#endif
