#ifndef FUNCTION_TABLE_H_
#define FUNCTION_TABLE_H_

// C/C++ headers
#include <vector>
#include <string>
#include <iostream>

namespace zhetapi {

	template <class T, class U>
	class Function;

	template <class T, class U>
	class ftable {
	public:
		struct node {
			Function <T, U>	__val;
			node *		__left;
			node *		__right;
		};
	private:
		node *		__tree;
		std::size_t	__size;
	public:
		ftable();
		ftable(const ftable &);

		ftable(const std::vector <Function <T, U>> &);

		template <class ... A>
		ftable(A ...);

		const ftable &operator=(const ftable &);

		~ftable();
		
		Function <T, U> &get(const std::string &);
		const Function <T, U> &find(const std::string &);

		bool contains(const std::string &);

		bool insert(const Function <T, U> &);

		bool remove(const Function <T, U> &);
		bool remove(const std::string &);

		bool empty() const;

		std::size_t size() const;

		void clear();

		void print() const;
	private:
		void gather(std::vector <Function <T, U>> &, Function <T, U>) const;
		
		template <class ... A>
		void gather(std::vector <Function <T, U>> &, Function <T, U>, A ...) const;

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
	ftable <T, U> ::ftable() : __tree(nullptr), __size(0) {}

	template <class T, class U>
	ftable <T, U> ::ftable(const ftable &other) : __tree(nullptr), __size(0)
	{
		*this = other;
	}

	template <class T, class U>
	ftable <T, U> ::ftable(const std::vector <Function <T, U>> &fs)
		: ftable()
	{
		for (auto ft : fs)
			insert(ft);
	}

	template <class T, class U>
	template <class ... A>
	ftable <T, U> ::ftable(A ... args) : ftable()
	{
		std::vector <Function <T, U>> pr;
		gather(pr, args...);
		
		for (auto ft : pr)
			insert(ft);
	}

	template <class T, class U>
	const ftable <T, U> &ftable <T, U> ::operator=(const ftable &other)
	{
		if (this != &other) {
			clear();

			__tree = clone(other.__tree);
			__size = other.__size;
		}

		return *this;
	}

	template <class T, class U>
	ftable <T, U> ::~ftable()
	{
		clear();
	}

	/* Public interface of ftable;
	* find methods, clear, print, etc. */
	template <class T, class U>
	Function <T, U> &ftable <T, U> ::get(const std::string &key)
	{
		if (!__tree)
			throw null_tree();

		splay(__tree, key);

		if (__tree->__val.symbol() != key)
			throw null_entry();

		return __tree->__val;
	}

	template <class T, class U>
	const Function <T, U> &ftable <T, U> ::find(const std::string &key)
	{
		if (!__tree)
			throw null_tree();

		splay(__tree, key);

		if (__tree->__val.symbol() != key)
			throw null_entry();

		return __tree->__val;
	}

	template <class T, class U>
	bool ftable <T, U> ::contains(const std::string &key)
	{
		Function <T, U> x;

		try {
			x = get(key);
		} catch (...) {
			return false;
		}

		return true;
	}

	template <class T, class U>
	bool ftable <T, U> ::insert(const Function <T, U> &x)
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
	bool ftable <T, U> ::remove(const Function <T, U> &x)
	{
		if (!__size)
			return false;

		splay(__tree, x.symbol());

		if (x.symbol() != __tree->__val.symbol())
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
	bool ftable <T, U> ::remove(const std::string &str)
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
	bool ftable <T, U> ::empty() const
	{
		return !__size;
	}

	template <class T, class U>
	std::size_t ftable <T, U> ::size() const
	{
		return size;
	}

	template <class T, class U>
	void ftable <T, U> ::clear()
	{
		if (__tree != nullptr)
			clear(__tree);
	}

	template <class T, class U>
	void ftable <T, U> ::print() const
	{
		print(__tree, 1, 0);
	}

	/* Protected methods (helpers methods);
	* splay, rotate left/right, clone, ect. */
	template <class T, class U>
	void ftable <T, U> ::gather(std::vector <Function <T, U>> &pr, Function <T, U> ft) const
	{
		pr.push_back(ft);
	}

	template <class T, class U>
	template <class ... A>
	void ftable <T, U> ::gather(std::vector <Function <T, U>> &pr, Function <T, U> ft, A ... args) const
	{
		pr.push_back(ft);
		gather(pr, args...);
	}

	template <class T, class U>
	typename ftable <T, U> ::node *ftable <T, U> ::clone(node *fnd)
	{
		node *nnode;

		if (fnd == nullptr)
			return nullptr;

		nnode = new node {fnd->__val, clone(fnd->__left),
			clone(fnd->__right)};

		return nnode;
	}

	template <class T, class U>
	void ftable <T, U> ::clear(node *(&fnd))
	{
		if (fnd == nullptr)
			return;

		clear(fnd->__left);
		clear(fnd->__right);

		delete fnd;
		
		// maybe remove later
		fnd = nullptr;
		__size--;
	}

	template <class T, class U>
	void ftable <T, U> ::print(node *fnd, int lev, int dir) const
	{
		if (fnd == nullptr)
			return;

		for (int i = 0; i < lev; i++)
			std::cout << "\t";

		if (!dir)
			std::cout << "Level #" << lev << " -- Root: ";
		else if (dir == 1)
			std::cout << "Level #" << lev << " -- Left: ";
		else
			std::cout << "Level #" << lev << " -- Right: ";

		if (fnd == nullptr)
			std::cout << "NULL";
		else
			std::cout << fnd->__val;
		
		std::cout << std::endl;
		
		if (fnd == nullptr)
			return;

		print(fnd->__left, lev + 1, 1);
		print(fnd->__right, lev + 1, -1);
	}

	template <class T, class U>
	void ftable <T, U> ::splay(node *(&fnd), const std::string &id)
	{
		node *rt = nullptr;
		node *lt = nullptr;
		node *rtm = nullptr;
		node *ltm = nullptr;
		
		while (fnd != nullptr) {
			if (id < fnd->__val.symbol()) {
				if (fnd->__left == nullptr)
					break;
				
				if (id < fnd->__left->__val.symbol()) {
					rotate_left(fnd);

					if (fnd->__left == nullptr)
						break;
				}

				if (rt == nullptr)
					rt = new node {Function <T, U>(), nullptr, nullptr};
			

				if (rtm == nullptr) {
					rt->__left = fnd;
					rtm = rt;
				} else {
					rtm->__left = fnd;
				}

				rtm = rtm->__left;
				
				fnd = rtm->__left;
				rtm->__left = nullptr;
			} else if (id > fnd->__val.symbol()) {
				if (fnd->__right == nullptr)
					break;
				
				if (id > fnd->__right->__val.symbol()) {
					rotate_right(fnd);
					if (fnd->__right == nullptr)
						break;
				}

				if (lt == nullptr)
					lt = new node {Function <T, U>(), nullptr, nullptr};

				if (ltm == nullptr) {
					lt->__right = fnd;
					ltm = lt;
				} else {
					ltm->__right = fnd;
				}

				ltm = ltm->__right;

				fnd = ltm->__right;
				ltm->__right = nullptr;
			} else {
				break;
			}
		}
		
		if (lt != nullptr) {
			ltm->__right = fnd->__left;
			fnd->__left = lt->__right;
		}

		if (rt != nullptr) {
			rtm->__left = fnd->__right;
			fnd->__right = rt->__left;
		}
	}

	template <class T, class U>
	void ftable <T, U> ::rotate_left(node *(&fnd))
	{
		node *rt = fnd->__left;

		fnd->__left = rt->__right;
		rt->__right = fnd;
		fnd = rt;
	}

	template <class T, class U>
	void ftable <T, U> ::rotate_right(typename ftable <T, U> ::node *(&fnd))
	{
		node *rt = fnd->__right;

		fnd->__right = rt->__left;
		rt->__left = fnd;
		fnd = rt;
	}

}

#endif
