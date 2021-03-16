#include <stack>

#include <core/node.hpp>
#include <core/operation_holder.hpp>

namespace zhetapi {

node::node() {}

node::node(const node &other)
		: __label(other.__label), __class(other.__class),
		__nodes(other.__nodes), __leaves(other.__leaves)
{
	if (other.__tptr)
		__tptr = other.__tptr->copy();
}

node::node(Token *tptr)
		: __tptr(tptr) {}

node::node(Token *tptr, lbl label)
		: __tptr(tptr), __label(label) {}

node::node(Token *tptr, const node &a, bool bl)
		: __leaves({a})
{
	if (tptr)
		__tptr = tptr->copy();
}

node::node(Token *tptr, const node &a, const node &b)
		: __leaves({a, b})
{
	if (tptr)
		__tptr = tptr->copy();
}

node::node(Token *tptr, const std::vector <node> &leaves)
		: __leaves(leaves)
{
	if (tptr)
		__tptr = tptr->copy();
}

node::node(Token *tptr, lbl label, const std::vector <node> &leaves)
		: __label(label), __leaves(leaves)
{
	if (tptr)
		__tptr = tptr->copy();
}

node &node::operator=(const node &other)
{
	if (this != &other) {
		// Separate clear() method?
		if (__tptr) {
			delete __tptr;

			__tptr = nullptr;
		}
		
		if (other.__tptr)
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
	
	__tptr = nullptr;
}

Token *node::copy_token() const
{
	if (__tptr)
		return __tptr->copy();

	return nullptr;
}

node &node::operator[](size_t i)
{
	return __leaves[i];
}

const node &node::operator[](size_t i) const
{
	return __leaves[i];
}

bool node::null() const
{
	return (__tptr == nullptr);
}

lbl node::label() const
{
	return __label;
}

bool node::empty() const
{
	return (__tptr == nullptr) && __leaves.empty();
}

Token *node::ptr() const
{
	return __tptr;
}

size_t node::child_count() const
{
	return __leaves.size();
}

Token::type node::caller() const
{
	if (__tptr)
		return __tptr->caller();

	return Token::undefined;
}

void node::transfer(const node &ref)
{
	if (ref.__tptr) {
		if (__tptr)
			delete __tptr;
		
		__tptr = ref.__tptr->copy();
	}

	__label = ref.__label;
	__class = ref.__class;
	__nodes = ref.__nodes;
	__leaves = ref.__leaves;
}

void node::append(const node &ref)
{
	__leaves.push_back(ref);
}

// Needs a better name
void node::append_front(const node &ref)
{
	__leaves.insert(__leaves.begin(), ref);
}

void node::print(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";

		counter--;
	}

	if (__tptr) {
		std::cout << "#" << num << ": " << __tptr->str()
			<< " (" << __tptr << ", " << strlabs[__label]
			<< ") @ " << this << std::endl;	
	} else {
		std::cout << "#" << num << ": null ("
			<< __tptr << ", " << strlabs[__label] << ") @ "
			<< this << std::endl;
	}

	counter = 0;
	for (node itr : __leaves)
		itr.print(++counter, lev + 1);
}

void node::print_no_address(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	if (__tptr) {
		std::cout << "#" << num << ": " << __tptr->str() << " (" <<
			strlabs[__label] << ") " << __nodes << " nodes" << ::std::endl;
	} else {
		std::cout << "#" << num << ": null (" <<
			strlabs[__label] << ") " << __nodes << " nodes" << ::std::endl;
	}

	counter = 0;
	for (node itr : __leaves)
		itr.print_no_address(++counter, lev + 1);
}

std::string node::display(int num, int lev) const
{
	std::ostringstream oss;

	oss << std::endl;

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
	if (!tokcmp(a.ptr(), b.ptr()))
		return false;
	
	// Check the leaves
	if (a.child_count() != b.child_count())
		return false;

	for (size_t i = 0; i < a.child_count(); i++) {
		if (!loose_match(a[i], b[i]))
			return false;
	}

	return true;
}

// Non-member functions
node factorize(const node &ref, const node &factor)
{
	using namespace std;
	// First check equality
	//
	// TODO: Create a nodecmp function (refactor loose_match)
	if (node::loose_match(ref, factor)) {
		cout << "\tRETURN UNIT" << endl;
		return node(new opd_z(1));
	}
	
	// Proceed normally
	std::vector <node> factors;
	std::stack <node> process;

	process.push(ref);

	node top;
	while (!process.empty()) {
		top = process.top();

		process.pop();

		operation_holder *ophptr = dynamic_cast <operation_holder *> (top.__tptr);

		if (ophptr && (ophptr->code == mul)) {
			process.push(top[0]);
			process.push(top[1]);
		} else {
			factors.push_back(top);
		}
	}

	using namespace std;
	cout << "Factors:" << endl;
	for (auto nd : factors)
		nd.print();
	
	std::vector <node> remaining;
	bool factorable = false;

	for (node nd : factors) {
		if (factorable) {
			remaining.push_back(nd);
			cout << "pushing" << endl;
			nd.print();

			continue;
		}

		if (node::loose_match(factor, nd)) {
			factorable = true;
		} else {
			remaining.push_back(nd);
			cout << "pushing" << endl;
			nd.print();
		}
	}

	if (!factorable)
		return node(nullptr);
	
	cout << "rem.size = " << remaining.size() << endl;

	// Fold into multiplication
	while (remaining.size() > 1) {
		std::vector <node> tmp;

		size_t n = remaining.size();

		for (size_t i = 0; i < n/2; i++) {
			tmp.push_back(
				node(new operation_holder("*"),
					{
						remaining[i],
						remaining[i + 1]
					}
				)
			);
		}

		if (n % 2)
			tmp.push_back(remaining[n - 1]);
	
		remaining = tmp;
	}

	cout << "RET rem.size = " << remaining.size() << endl;
	remaining[0].print();

	return remaining[0];
}

std::ostream &operator<<(std::ostream &os, const node &tree)
{
	os << tree.display();

	return os;
}

}
