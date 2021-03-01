#include <core/node.hpp>

namespace zhetapi {

node::node() {}

node::node(const node &other)
		: __tptr(other.__tptr->copy()), __label(other.__label),
		__class(other.__class), __nodes(other.__nodes),
		__leaves(other.__leaves) {}

node::node(Token *tptr, const node &a, bool bl)
		: __tptr(tptr->copy()), __leaves({a}) {}

node::node(Token *tptr, const node &a, const node &b)
		: __tptr(tptr->copy()), __leaves({a, b}) {}

node::node(Token *tptr, const std::vector <node> &leaves)
		: __tptr(tptr->copy()), __leaves(leaves) {}

node::node(Token *tptr, lbl label, const std::vector <node> &leaves)
		: __tptr(tptr->copy()), __label(label), __leaves(leaves) {}

node &node::operator=(const node &other)
{
	if (this != &other) {
		// Separate clear() method?
		if (__tptr)
			delete __tptr;
		
		__tptr = other.__tptr->copy();
		__label = other.__label;
		__class = other.__class;
		__nodes = other.__nodes;
		__leaves = other.__leaves;
	}

	return *this;
}

node::~node()
{
	if (__tptr)
		delete __tptr;
}

bool node::empty() const
{
	return (__tptr == nullptr) && __leaves.empty();
}

void node::transfer(const node &ref)
{
	__tptr = ref.__tptr;
	__label = ref.__label;
	__class = ref.__class;
	__leaves = ref.__leaves;
}

void node::append(const node &ref)
{
	__leaves.push_back(ref);
}

void node::print(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		::std::cout << "\t";
		counter--;
	}

	if (__tptr) {
		::std::cout << "#" << num << ": " << __tptr->str() << " (" <<
			__tptr << ", " << strlabs[__label] << ") @ " << this <<
			::std::endl;	
	} else {
		::std::cout << "#" << num << ": null (" <<
			__tptr << ", " << strlabs[__label] << ") @ " << this <<
			::std::endl;
	}

	counter = 0;
	for (node itr : __leaves)
		itr.print(++counter, lev + 1);
}

void node::print_no_address(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		::std::cout << "\t";
		counter--;
	}

	if (__tptr) {
		::std::cout << "#" << num << ": " << __tptr->str() << " (" <<
			strlabs[__label] << ") " << __nodes << " nodes" << ::std::endl;
	} else {
		::std::cout << "#" << num << ": null (" <<
			strlabs[__label] << ") " << __nodes << " nodes" << ::std::endl;
	}

	counter = 0;
	for (node itr : __leaves)
		itr.print_no_address(++counter, lev + 1);
}

::std::string node::display(int num, int lev) const
{
	::std::ostringstream oss;

	oss << ::std::endl;

	int counter = lev;
	while (counter > 0) {
		oss << "\t";
		counter--;
	}

	oss << "#" << num << ": " << __tptr->str() << " (" << __tptr << ") @ "
		<< this << " "  << __nodes << " nodes" << ::std::endl;

	counter = 0;
	for (node itr : __leaves)
		oss << itr.display(++counter, lev + 1);

	return oss.str();
}

bool node::loose_match(const node &a, const node &b)
{
	// Check the Token
	if (*(a.__tptr) != b.__tptr)
		return false;
	
	// Check the leaves
	if (a.__leaves.size() != b.__leaves.size())
		return false;

	for (size_t i = 0; i < a.__leaves.size(); i++) {
		if (!loose_match(a.__leaves[i], b.__leaves[i]))
			return false;
	}

	return true;
}

::std::ostream &operator<<(::std::ostream &os, const node &tree)
{
	os << tree.display();

	return os;
}

}
