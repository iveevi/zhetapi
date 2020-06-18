#ifndef VTABLE_H_
#define VTABLE_H_

template <class T>
class variable;

template <class T>
class vtable {
public:
	using var = variable <T>;
	
	struct node {
		var val;
		node *left;
		node *right;
	};

private:
	node *tree;

	std::size_t sz;
public:
	vtable();
	vtable(const vtable &);

	vtable(const std::vector <var> &);

	template <class ... U>
	vtable(U ...);

	const vtable &operator=(const vtable &);

	~vtable();

	var &get(const std::string &);
	const var &find(const std::string &);
	
	bool insert(const var &);
	
	bool remove(const var &);
	bool remove(const std::string &);

	bool empty() const;

	std::size_t size() const;

	void clear();

	void print() const;
private:
	void gather(std::vector <var> &, var) const;
	
	template <class ... U>
	void gather(std::vector <var> &, var, U ...) const;

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
template <class T>
vtable <T> ::vtable() : tree(nullptr), sz(0) {}

template <class T>
vtable <T> ::vtable(const vtable &other) : tree(nullptr), sz(0)
{
	*this = other;
}

template <class T>
vtable <T> ::vtable(const std::vector <var> &vs) : vtable()
{
	for (auto vr : vs)
		insert(vr);
}

template <class T>
template <class ... U>
vtable <T> ::vtable(U ... args) : vtable()
{
	std::vector <var> pr;
	gather(pr, args...);

	for (auto vr : pr.first)
		insert(vr);
}

template <class T>
const vtable <T> &vtable <T> ::operator=(const vtable &other)
{
	if (this != &other) {
		clear();
		tree = clone(other.tree);
		sz = other.sz;
	}

	// cout << string(50, '_') << endl;
	// cout << "ASSIGNMENT OPERATOR:" << endl;
	// print();

	return *this;
}

template <class T>
vtable <T> ::~vtable()
{
	clear();
}

/* Public interface of vtable;
 * find methods, clear, print, etc. */
template <class T>
typename vtable <T> ::var &vtable <T> ::get(const std::string &key)
{
	if (!tree)
		throw null_tree();

	splay(tree, key);

	if (tree->val.symbol() != key)
		throw null_entry();

	return tree->val;
}

template <class T>
const typename vtable <T> ::var &vtable <T> ::find(const std::string &key)
{
	if (!tree)
		throw null_tree();

	splay(tree, key);

	if (tree->val.symbol() != key)
		throw null_entry();

	return tree->val;
}

template <class T>
bool vtable <T> ::insert(const var &x)
{
	if (tree == nullptr) {
		tree = new node {x, nullptr, nullptr};
		sz++;
		return true;
	}

	splay(tree, x.symbol());

	node *temp = new node {x, nullptr, nullptr};

	if (x < tree->val) {
		temp->left = tree->left;
		temp->right = tree;

		temp->right->left = nullptr;
		tree = temp;
		sz++;

		return true;
	}

	if (x > tree->val) {
		temp->left = tree;
		temp->right = tree->right;

		temp->left->right = nullptr;
		tree = temp;
		sz++;

		return true;
	}

	return false;
}

template <class T>
bool vtable <T> ::remove(const var &x)
{
	if (!sz)
		return false;

	splay(tree, x.symbol());

	if (x != tree->val)
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
	sz--;

	return true;
}

template <class T>
bool vtable <T> ::remove(const std::string &str)
{
	if (!sz)
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
	sz--;

	return true;
}

template <class T>
bool vtable <T> ::empty() const
{
	return !sz;
}

template <class T>
std::size_t vtable <T> ::size() const
{
	return sz;
}

template <class T>
void vtable <T> ::clear()
{
	if (tree != nullptr)
		clear(tree);
}

template <class T>
void vtable <T> ::print() const
{
	print(tree, 1, 0);
}

/* Protected methods (helpers methods);
 * splay, rotate left/right, clone, ect. */
template <class T>
void vtable <T> ::gather(std::vector <var> &pr, var vs) const
{
	pr.push_back(vs);
}

template <class T>
template <class ... U>
void vtable <T> ::gather(std::vector <var> &pr, var vs, U ... args) const
{
	pr.push_back(vs);
	gather(pr, args...);
}

template <class T>
typename vtable <T> ::node *vtable <T> ::clone(node *vnd)
{
	node *nnode;

	if (vnd == nullptr)
		return nullptr;

	nnode = new node {vnd->val, clone(vnd->left),
		clone(vnd->right)};

	return nnode;
}

template <class T>
void vtable <T> ::clear(node *(&vnd))
{
	if (vnd == nullptr)
		return;

	clear(vnd->left);
	clear(vnd->right);

	delete vnd;
	
	// maybe remove later
	vnd = nullptr;
	sz--;
}

template <class T>
void vtable <T> ::print(node *vnd, int lev, int dir) const
{
	if (vnd == nullptr)
		return;

	for (int i = 0; i < lev; i++)
		std::cout << "\t";

	switch (dir) {
	case 0:
		std::cout << "Level #" << lev << " -- Root: ";

		if (vnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << vnd->val;
		std::cout << " [@" << vnd << "]" << std::endl;
		break;
	case 1:
		std::cout << "Level #" << lev << " -- Left: ";
		
		if (vnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << vnd->val;
		std::cout << " [@" << vnd << "]" << std::endl;
		break;
	case -1:
		std::cout << "Level #" << lev << " -- Right: ";
		
		if (vnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << vnd->val;
		std::cout << " [@" << vnd << "]" << std::endl;
		break;
	}
	
	if (vnd == nullptr)
		return;

	print(vnd->left, lev + 1, 1);
	print(vnd->right, lev + 1, -1);
}

template <class T>
void vtable <T> ::splay(node *(&vnd), const std::string &id)
{
	node *rt = nullptr;
	node *lt = nullptr;
	node *rtm = nullptr;
	node *ltm = nullptr;
	
	while (vnd != nullptr) {
		if (id < vnd->val.symbol()) {
			if (vnd->left == nullptr)
				break;
			
			if (id < vnd->left->val.symbol()) {
				rotate_left(vnd);

				if (vnd->left == nullptr)
					break;
			}

			if (rt == nullptr)
				rt = new node {var(), nullptr, nullptr};
		

			if (rtm == nullptr) {
				rt->left = vnd;
				rtm = rt;
			} else {
				rtm->left = vnd;
			}

			rtm = rtm->left;
			
			vnd = rtm->left;
			rtm->left = nullptr;
		} else if (id > vnd->val.symbol()) {
			if (vnd->right == nullptr)
				break;
			
			if (id > vnd->right->val.symbol()) {
				rotate_right(vnd);
				if (vnd->right == nullptr)
					break;
			}

			if (lt == nullptr)
				lt = new node {var(), nullptr, nullptr};

			if (ltm == nullptr) {
				lt->right = vnd;
				ltm = lt;
			} else {
				ltm->right = vnd;
			}

			ltm = ltm->right;

			vnd = ltm->right;
			ltm->right = nullptr;
		} else {
			break;
		}
	}
	
	if (lt != nullptr) {
		ltm->right = vnd->left;
		vnd->left = lt->right;
	}

	if (rt != nullptr) {
		rtm->left = vnd->right;
		vnd->right = rt->left;
	}
}

template <class T>
void vtable <T> ::rotate_left(node *(&vnd))
{
	node *rt = vnd->left;

	vnd->left = rt->right;
	rt->right = vnd;
	vnd = rt;
}

template <class T>
void vtable <T> ::rotate_right(node *(&vnd))
{
	node *rt = vnd->right;

	vnd->right = rt->left;
	rt->left = vnd;
	vnd = rt;
}

#endif
