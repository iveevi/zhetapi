#ifndef VTABLE_H_
#define VTABLE_H_

// C/C++ headers
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

namespace zhetapi {

	template <class T, class U>
	class Variable;

	template <class T, class U>
	class vtable {
	public:
		struct node {
			Variable <T, U>	__val;
			node		*__left;
			node		*__right;
		};

	private:
		node		*__tree;
		std::size_t	__size;
	public:
		vtable();
		vtable(const vtable &);

		vtable(const std::vector <Variable <T, U>> &);

		template <class ... A>
		vtable(A ...);

		const vtable &operator=(const vtable &);

		~vtable();

		Variable <T, U> &get(const std::string &);
		const Variable <T, U> &find(const std::string &);

		std::vector <Variable <T, U>> list() const;

		bool contains(const std::string &);
		
		bool insert(const Variable <T, U>&);
		
		bool remove(const Variable <T, U>&);
		bool remove(const std::string &);

		bool empty() const;

		std::size_t size() const;

		void clear();

		void print() const;
	private:
		void gather(std::vector <Variable <T, U>> &, Variable <T, U>) const;
		
		template <class ... A>
		void gather(std::vector <Variable <T, U>> &, Variable <T, U>, A ...) const;

		node *clone(node *);

		void clear(node *(&));

		void list(node *, std::vector <Variable <T, U>> &) const;

		void print(node *, int, int) const;

		void splay(node *(&), const std::string &);

		void rotate_left(node *(&));
		void rotate_right(node *(&));
	public:
		// Empty or null root nodes
		class null_tree {};

		// Node not found
		class null_entry {};
	};

	/* Constructors, destructors,
	 * major/significant operators */
	template <class T, class U>
	vtable <T, U> ::vtable() : __tree(nullptr), __size(0) {}

	template <class T, class U>
	vtable <T, U> ::vtable(const vtable &other) : __tree(nullptr), __size(0)
	{
		*this = other;
	}

	template <class T, class U>
	vtable <T, U> ::vtable(const std::vector <Variable <T, U>> &vs) : vtable()
	{
		for (auto vr : vs)
			insert(vr);
	}

	template <class T, class U>
	template <class ... A>
	vtable <T, U> ::vtable(A ... args) : vtable()
	{
		std::vector <Variable <T, U>> pr;

		gather(pr, args...);

		for (auto vr : pr.first)
			insert(vr);
	}

	template <class T, class U>
	const vtable <T, U> &vtable <T, U> ::operator=(const vtable &other)
	{
		if (this != &other) {
			clear();
			__tree = clone(other.__tree);
			__size = other.__size;
		}

		return *this;
	}

	template <class T, class U>
	vtable <T, U> ::~vtable()
	{
		clear();
	}

	template <class T, class U>
	Variable <T, U> &vtable <T, U> ::get(const std::string &key)
	{
		if (!__tree)
			throw null_tree();

		splay(__tree, key);

		if (__tree->__val.symbol() != key)
			throw null_entry();

		return __tree->__val;
	}

	template <class T, class U>
	const Variable <T, U> &vtable <T, U> ::find(const std::string &key)
	{
		if (!__tree)
			throw null_tree();

		splay(__tree, key);

		if (__tree->__val.symbol() != key)
			throw null_entry();

		return __tree->__val;
	}

	template <class T, class U>
	std::vector <Variable <T, U>> vtable <T, U> ::list() const
	{
		std::vector <Variable <T, U>> v;

		if (__tree)
			list(__tree, v);

		return v;
	}

	template <class T, class U>
	bool vtable <T, U> ::contains(const std::string &key)
	{
		Variable <T, U> x;

		try {
			x = get(key);
		} catch (...) {
			return false;
		}

		return true;
	}

	template <class T, class U>
	bool vtable <T, U> ::insert(const Variable <T, U> &x)
	{
		if (__tree == nullptr) {
			__tree = new node {x, nullptr, nullptr};
			__size++;
			return true;
		}

		splay(__tree, x.symbol());

		node *temp = new node {x, nullptr, nullptr};

		if (x < __tree->__val) {
			temp->__left = __tree->__left;
			temp->__right = __tree;

			temp->__right->__left = nullptr;
			__tree = temp;
			__size++;

			return true;
		}

		if (x > __tree->__val) {			
			temp->__left = __tree;
			temp->__right = __tree->__right;

			temp->__left->__right = nullptr;
			__tree = temp;
			__size++;

			return true;
		}

		return false;
	}

	template <class T, class U>
	bool vtable <T, U> ::remove(const Variable <T, U>&x)
	{
		if (!__size)
			return false;

		splay(__tree, x.symbol());

		if (x != __tree->__val)
			return false;

		node *nnd;
		if (__tree->__left == nullptr) {
			nnd = __tree->__left;
		} else {
			nnd = __tree->__left;
			splay(nnd, x.symbol());

			nnd->__right = __tree->__right;
		}

		delete __tree;

		__tree = nnd;
		__size--;

		return true;
	}

	template <class T, class U>
	bool vtable <T, U> ::remove(const std::string &str)
	{
		if (!__size)
			return false;

		splay(__tree, str);

		if (str != __tree->__val.symbol())
			return false;

		node *nnd;
		if (__tree->__left == nullptr) {
			nnd = __tree->__left;
		} else {
			nnd = __tree->__left;

			splay(nnd, str);

			nnd->__right = __tree->__right;
		}

		delete __tree;

		__tree = nnd;
		__size--;

		return true;
	}

	template <class T, class U>
	bool vtable <T, U> ::empty() const
	{
		return !__size;
	}

	template <class T, class U>
	std::size_t vtable <T, U> ::size() const
	{
		return __size;
	}

	template <class T, class U>
	void vtable <T, U> ::clear()
	{
		if (__tree != nullptr)
			clear(__tree);
	}

	template <class T, class U>
	void vtable <T, U> ::print() const
	{
		print(__tree, 1, 0);
	}

	/* Protected methods (helpers methods);
	 * splay, rotate left/right, clone, ect. */
	template <class T, class U>
	void vtable <T, U> ::gather(std::vector <Variable <T, U>> &pr, Variable <T, U>vs) const
	{
		pr.push_back(vs);
	}

	template <class T, class U>
	template <class ... A>
	void vtable <T, U> ::gather(std::vector <Variable <T, U>> &pr, Variable <T, U> vs, A ... args) const
	{
		pr.push_back(vs);

		gather(pr, args...);
	}

	template <class T, class U>
	typename vtable <T, U> ::node *vtable <T, U> ::clone(node *vnd)
	{
		node *nnode;

		if (vnd == nullptr)
			return nullptr;

		nnode = new node {vnd->__val, clone(vnd->__left),
			clone(vnd->__right)};

		return nnode;
	}

	template <class T, class U>
	void vtable <T, U> ::clear(node *(&vnd))
	{
		if (vnd == nullptr)
			return;

		clear(vnd->__left);
		clear(vnd->__right);

		delete vnd;
		
		// maybe remove later
		vnd = nullptr;

		__size--;
	}

	template <class T, class U>
	void vtable <T, U> ::list(node *ref, std::vector <Variable <T, U>> &v) const
	{
		v.push_back(ref->__val);

		if (ref->__left)
			list(ref->__left, v);
		
		if (ref->__right)
			list(ref->__right, v);
	}

	template <class T, class U>
	void vtable <T, U> ::print(node *vnd, int lev, int dir) const
	{
		if (vnd == nullptr)
			return;

		for (int i = 0; i < lev; i++)
			std::cout << "\t";

		if (!dir)
			std::cout << "Level #" << lev << " -- Root: ";
		else if (dir == 1)
			std::cout << "Level #" << lev << " -- Left: ";
		else
			std::cout << "Level #" << lev << " -- Right: ";

		if (vnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << vnd->__val;

		std::cout << std::endl;
		
		if (vnd == nullptr)
			return;

		print(vnd->__left, lev + 1, 1);
		print(vnd->__right, lev + 1, -1);
	}

	template <class T, class U>
	void vtable <T, U> ::splay(node *(&vnd), const std::string &id)
	{
		node *rt = nullptr;
		node *lt = nullptr;
		node *rtm = nullptr;
		node *ltm = nullptr;
		
		while (vnd != nullptr) {
			if (id < vnd->__val.symbol()) {
				if (vnd->__left == nullptr)
					break;
				
				if (id < vnd->__left->__val.symbol()) {
					rotate_left(vnd);

					if (vnd->__left == nullptr)
						break;
				}

				if (rt == nullptr)
					rt = new node {Variable <T, U> (), nullptr, nullptr};
			

				if (rtm == nullptr) {
					rt->__left = vnd;
					rtm = rt;
				} else {
					rtm->__left = vnd;
				}

				rtm = rtm->__left;
				
				vnd = rtm->__left;
				rtm->__left = nullptr;
			} else if (id > vnd->__val.symbol()) {
				if (vnd->__right == nullptr)
					break;
				
				if (id > vnd->__right->__val.symbol()) {
					rotate_right(vnd);
					if (vnd->__right == nullptr)
						break;
				}

				if (lt == nullptr)
					lt = new node {Variable <T, U> (), nullptr, nullptr};

				if (ltm == nullptr) {
					lt->__right = vnd;
					ltm = lt;
				} else {
					ltm->__right = vnd;
				}

				ltm = ltm->__right;

				vnd = ltm->__right;
				ltm->__right = nullptr;
			} else {
				break;
			}
		}
		
		if (lt != nullptr) {
			ltm->__right = vnd->__left;
			vnd->__left = lt->__right;
		}

		if (rt != nullptr) {
			rtm->__left = vnd->__right;
			vnd->__right = rt->__left;
		}
	}

	template <class T, class U>
	void vtable <T, U> ::rotate_left(node *(&vnd))
	{
		node *rt = vnd->__left;

		vnd->__left = rt->__right;
		rt->__right = vnd;
		vnd = rt;
	}

	template <class T, class U>
	void vtable <T, U> ::rotate_right(node *(&vnd))
	{
		node *rt = vnd->__right;

		vnd->__right = rt->__left;
		rt->__left = vnd;
		vnd = rt;
	}

}

#endif
