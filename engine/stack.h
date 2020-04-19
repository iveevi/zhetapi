#ifndef STACK_H_
#define STACK_H_

#include <iostream>

// remove later
class not_found_exception {};
class null_tree_exception {};
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
	const T &find(const T &);

	bool empty() const;
	std::size_t size() const;

	void clear();

	const splay_stack &operator= (const splay_stack &);

	bool insert(const T &);
	bool remove(const T &);

	void print() const;
protected:
	node *clone(node *);

	void clear(node *(&));
	void print(node *, int, int) const;

	void rotate_left(node *(&));
	void rotate_right(node *(&));

	// overload this in derived classes
	virtual void splay(node *(&), const T &);

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
const T &splay_stack <T> ::find(const T &key)
{
	if (!m_root)
		throw null_tree_exception();

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
	if (this != &rhs) {
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
		temp->right = m_root;

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

	node *nnd;
	if (m_root->left == nullptr) {
		nnd = m_root->left;
	} else {
		nnd = m_root->left;
		splay(nnd, val);

		nnd->right = m_root->right;
	}

	delete m_root;

	m_root = nnd;
	m_size--;

	return true;
}

template <class T>
void splay_stack <T> ::print() const
{
	print(m_root, 0, 0);
}

/* splay_tree other member
 * functions (protected interface) */
template <class T>
typename splay_stack <T> ::node *splay_stack <T>
	::clone(node *nd)
{
	node *nnode;

	if (nd == nullptr)
		return nullptr;

	nnode = new node {nd->val, clone(nd->left),
		clone(nd->right)};

	return nnode;
}

template <class T>
void splay_stack <T> ::clear(node *(&nd))
{
	if (nd == nullptr)
		return;

	clear(nd->left);
	clear(nd->right);

	delete nd;
	
	// maybe remove later
	nd = nullptr;
	m_size--;
}

template <class T>
void splay_stack <T> ::print(node *nd, int lev, int dir) const
{
	if (nd == nullptr) return;

	for (int i = 0; i < lev; i++)
		std::cout << "\t";

	switch (dir) {
	case 0:
		std::cout << "Level #" << lev << " -- Root: ";

		if (nd == nullptr)
			std::cout << "NULL";
		else
			std::cout << nd->val;
		std::cout << std::endl;;
		break;
	case 1:
		std::cout << "Level #" << lev << " -- Left: ";
		
		if (nd == nullptr)
			std::cout << "NULL";
		else
			std::cout << nd->val;
		std::cout << std::endl;;
		break;
	case -1:
		std::cout << "Level #" << lev << " -- Right: ";
		
		if (nd == nullptr)
			std::cout << "NULL";
		else
			std::cout << nd->val;
		std::cout << std::endl;;
		break;
	}
	
	if (nd == nullptr)
		return;

	print(nd->left, lev + 1, 1);
	print(nd->right, lev + 1, -1);
}

template <class T>
void splay_stack <T> ::rotate_left(node *(&nd))
{
	node *rt = nd->left;

	nd->left = rt->right;
	rt->right = nd;
	nd = rt;
}

template <class T>
void splay_stack <T> ::rotate_right(node *(&nd))
{
	node *rt = nd->right;

	nd->right = rt->left;
	rt->left = nd;
	nd = rt;
}

template <class T>
void splay_stack <T> ::splay(node *(&nd), const T &val)
{
	node *rt = nullptr;
	node *lt = nullptr;
	node *rtm = nullptr;
	node *ltm = nullptr;
	
	while (nd != nullptr) {
		if (val < nd->val) {
			if (nd->left == nullptr)
				break;
			
			if (val < nd->left->val) {
				rotate_left(nd);

				if (nd->left == nullptr)
					break;
			}

			if (rt == nullptr)
				rt = new node {T(), nullptr, nullptr};
		

			if (rtm == nullptr) {
				rt->left = nd;
				rtm = rt;
			} else {
				rtm->left = nd;
			}

			rtm = rtm->left;
			
			nd = rtm->left;
			rtm->left = nullptr;
		} else if (val > nd->val) {
			if (nd->right == nullptr)
				break;
			
			if (val > nd->right->val) {
				rotate_right(nd);
				if (nd->right == nullptr)
					break;
			}

			if (lt == nullptr)
				lt = new node {T(), nullptr, nullptr};

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
