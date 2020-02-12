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
	node *find(node *, const T &) const;

	bool insert(node *(&), const T &);
	bool remove(node *(&), const T &);

	void clear(node *(&));

	void rotate_left(node *(&));
	void rotate_right(node *(&));

	// overload this in derived classes
	virtual void splay(node *(&), const T &);
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
const T &splay_tree <T> ::find(const T &key) const
{
	splay(m_root, key);

	if (m_root->val != key)
		throw not_found_exception();

	return m_root->val;
}

template <class T>
bool splay_tree <T> ::empty() const
{
	return m_size == 0;
}

template <class T>
std::size_t splay_tree <T> ::size() const
{
	return m_size;
}

template <class T>
void splay_tree <T> ::clear()
{
	clear(m_root);
}

#endif
