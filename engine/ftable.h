#ifndef TABLE_H_
#define TABLE_H_

#include "node.h"
#include "functor.h"
#include "variable.h"

template <class T, class U>
class functor;

template <class T, class U>
class ftable {
public:
	using ftr = functor <T, U>;

	struct node {
		ftr val;
		node *left;
		node *right;
	};
private:
	node *tree;

	std::size_t size;
public:
	ftable();
	ftable(const ftable &);

	ftable(const std::vector <var> &, const std::vector <ftr> &);
	ftable(const std::pair <std::vector <var>, std::vector <ftr>> &);

	template <class ... U>
	ftable(U ...);

	const ftable &operator=(const ftable &);

	~ftable();
	
	ftr &get(const std::string &);
	const ftr &find(const std::string &);

	bool insert(const ftr &);

	bool remove(const ftr &);
	bool remove(const std::string &);

	bool empty() const;

	std::size_t size() const;

	void clear();

	void print() const;
private:
	void gather(std::vector <ftr> &, ftr) const;
	
	template <class ... U>
	void gather(std::vector <ftr> &, ftr, U ...) const;

	node *clone(node *);

	void clear(node *(&));

	void print(node *, int, int) const;

	void splay(node *(&), const std::string &);
	
	void rotate_left(node *(&));
	void rotate_right(node *(&));
public:
	// empty or null root nodes
	class null_tree {};

	// old nfe equivalent
	class null_entry {};
};

/* Constructors, destructors,
 * major/significant operators */
template <class T, class U>
ftable <T> ::ftable() : tree(nullptr), size(0) {}

template <class T, class U>
ftable <T> ::ftable(const ftable &other) : tree(nullptr), size(0)
{
	*this = other;
}

template <class T, class U>
ftable <T> ::ftable(const std::vector <ftr> &fs)
	: ftable()
{
	for (auto ft : fs)
		insert(ft);
}

template <class T, class U>
template <class ... U>
ftable <T> ::ftable(U ... args) : ftable()
{
	std::vector <ftr> pr;
	gather(pr, args...);
	
	for (auto ft : pr)
		insert(ft);
}

template <class T, class U>
const ftable <T> &ftable <T> ::operator=(const ftable &other)
{
	if (this != &other) {
		clear();
		vtree = clone_var(other.vtree);
		tree = clone(other.tree);
		vtree_size = other.vtree_size;
		size = other.size;
	}

	// cout << string(50, '_') << endl;
	// cout << "ASSIGNMENT OPERATOR:" << endl;
	// print();

	return *this;
}

template <class T, class U>
ftable <T> ::~ftable()
{
	clear();
}

/* Public interface of ftable;
 * find methods, clear, print, etc. */
template <class T, class U>
typename ftable <T> ::ftr &ftable <T> ::get(const std::string &key)
{
	if (!tree)
		throw null_tree();

	splay(tree, key);

	if (tree->val.symbol() != key)
		throw null_entry();

	return tree->val;
}

template <class T, class U>
const typename ftable <T> ::ftr &ftable <T> ::find(const std::string &key)
{
	if (!tree)
		throw null_tree();

	splay(tree, key);

	if (tree->val.symbol() != key)
		throw null_entry();

	return tree->val;
}

template <class T, class U>
bool ftable <T> ::insert(const ftr &x)
{
	if (tree == nullptr) {
		tree = new node {x, nullptr, nullptr};
		size++;
		return true;
	}

	splay(tree, x.symbol());

	node *temp = new node {x, nullptr, nullptr};

	if (x < tree->val) {
		temp->left = tree->left;
		temp->right = tree;

		temp->right->left = nullptr;
		tree = temp;
		size++;

		return true;
	}

	if (x > tree->val) {
		temp->left = tree;
		temp->right = tree->right;

		temp->left->right = nullptr;
		tree = temp;
		size++;

		return true;
	}

	return false;
}

template <class T, class U>
bool ftable <T> ::remove(const ftr &x)
{
	if (!size)
		return false;

	splay(tree, x.symbol());

	if (x.symbol() != tree->val.symbol())
		return false;

	node *nnd;
	if (tree->left == nullptr) {
		nnd = tree->left;
	} else {
		nnd = tree->left;
		splay(nnd, x.symbol());

		nnd->right = tree->right;
	}

	delete tree;

	tree = nnd;
	size--;

	return true;
}

template <class T, class U>
bool ftable <T> ::remove(const std::string &str)
{
	if (!size)
		return false;

	splay(tree, str);

	if (str != tree->val.symbol())
		return false;

	node *nnd;
	if (tree->left == nullptr) {
		nnd = tree->left;
	} else {
		nnd = tree->left;
		splay(nnd, str);

		nnd->right = tree->right;
	}

	delete tree;

	tree = nnd;
	size--;

	return true;
}

template <class T, class U>
bool ftable <T> ::empty() const
{
	return !size;
}

template <class T, class U>
std::size_t ftable <T> ::size() const
{
	return size;
}

template <class T, class U>
void ftable <T> ::clear()
{
	if (tree != nullptr)
		clear(tree);
}

template <class T, class U>
void ftable <T> ::print() const
{
	print(tree);
}

/* Protected methods (helpers methods);
 * splay, rotate left/right, clone, ect. */
template <class T, class U>
void ftable <T> ::gather(std::vector <ftr> &pr, ftr ft) const
{
	pr.push_back(ft);
}

template <class T, class U>
template <class ... U>
void ftable <T> ::gather(std::vector <ftr> &pr, ftr ft, U ... args) const
{
	pr.push_back(ft);
	gather(pr, args...);
}

template <class T, class U>
typename ftable <T> ::node *ftable <T> ::clone(node *fnd)
{
	node *nnode;

	if (fnd == nullptr)
		return nullptr;

	nnode = new node {fnd->val, clone(fnd->left),
		clone(fnd->right)};

	return nnode;
}

template <class T, class U>
void ftable <T> ::clear(node *(&fnd))
{
	if (fnd == nullptr)
		return;

	clear(fnd->left);
	clear(fnd->right);

	delete fnd;
	
	// maybe remove later
	fnd = nullptr;
	size--;
}

template <class T, class U>
void ftable <T> ::print(node *fnd, int lev, int dir) const
{
	if (fnd == nullptr)
		return;

	for (int i = 0; i < lev; i++)
		std::cout << "\t";

	switch (dir) {
	case 0:
		std::cout << "Level #" << lev << " -- Root: ";

		if (fnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << fnd->val;
		std::cout << " [@" << fnd << "]" << std::endl;
		break;
	case 1:
		std::cout << "Level #" << lev << " -- Left: ";
		
		if (fnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << fnd->val;
		std::cout << " [@" << fnd << "]" << std::endl;
		break;
	case -1:
		std::cout << "Level #" << lev << " -- Right: ";
		
		if (fnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << fnd->val;
		std::cout << " [@" << fnd << "]" << std::endl;
		break;
	}
	
	if (fnd == nullptr)
		return;

	print(fnd->left, lev + 1, 1);
	print(fnd->right, lev + 1, -1);
}

template <class T, class U>
void ftable <T> ::splay(node *(&fnd), const std::string &id)
{
	node *rt = nullptr;
	node *lt = nullptr;
	node *rtm = nullptr;
	node *ltm = nullptr;
	
	while (fnd != nullptr) {
		if (id < fnd->val.symbol()) {
			if (fnd->left == nullptr)
				break;
			
			if (id < fnd->left->val.symbol()) {
				rotate_left(fnd);

				if (fnd->left == nullptr)
					break;
			}

			if (rt == nullptr)
				rt = new node {ftr(), nullptr, nullptr};
		

			if (rtm == nullptr) {
				rt->left = fnd;
				rtm = rt;
			} else {
				rtm->left = fnd;
			}

			rtm = rtm->left;
			
			fnd = rtm->left;
			rtm->left = nullptr;
		} else if (id > fnd->val.symbol()) {
			if (fnd->right == nullptr)
				break;
			
			if (id > fnd->right->val.symbol()) {
				rotate_right(fnd);
				if (fnd->right == nullptr)
					break;
			}

			if (lt == nullptr)
				lt = new node {ftr(), nullptr, nullptr};

			if (ltm == nullptr) {
				lt->right = fnd;
				ltm = lt;
			} else {
				ltm->right = fnd;
			}

			ltm = ltm->right;

			fnd = ltm->right;
			ltm->right = nullptr;
		} else {
			break;
		}
	}
	
	if (lt != nullptr) {
		ltm->right = fnd->left;
		fnd->left = lt->right;
	}

	if (rt != nullptr) {
		rtm->left = fnd->right;
		fnd->right = rt->left;
	}
}

template <class T, class U>
void ftable <T> ::rotate_left(node *(&fnd))
{
	node *rt = fnd->left;

	fnd->left = rt->right;
	rt->right = fnd;
	fnd = rt;
}

template <class T, class U>
void ftable <T> ::rotate_right(node *(&fnd))
{
	node *rt = fnd->right;

	fnd->right = rt->left;
	rt->left = fnd;
	fnd = rt;
}

#endif
