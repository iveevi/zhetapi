#include <stack>

#include "../../engine/core/node.hpp"
#include "../../engine/core/operation_holder.hpp"
#include "../../engine/core/node_reference.hpp"

namespace zhetapi {

node::node() {}

node::node(const node &other)
		: _label(other._label), _class(other._class),
		_nodes(other._nodes), _leaves(other._leaves)
{
	if (other._tptr)
		_tptr = other._tptr->copy();
}

node::node(Token *tptr)
		: _tptr(tptr) {}

node::node(Token *tptr, lbl label)
		: _tptr(tptr), _label(label) {}

// TODO: Remove this bool
node::node(Token *tptr, const node &a)
		: _leaves({a})
{
	if (tptr)
		_tptr = tptr->copy();
}

node::node(Token *tptr, const node &a, const node &b)
		: _leaves({a, b})
{
	if (tptr)
		_tptr = tptr->copy();
}

node::node(Token *tptr, const std::vector <node> &leaves)
		: _leaves(leaves)
{
	if (tptr)
		_tptr = tptr->copy();
}

node::node(Token *tptr, lbl label, const std::vector <node> &leaves)
		: _label(label), _leaves(leaves)
{
	if (tptr)
		_tptr = tptr->copy();
}

node::node(Token *tptr, lbl label, const node &n1, const node &n2)
		: node(tptr, label, {n1, n2}) {}

node &node::operator=(const node &other)
{
	if (this != &other) {
		// Separate clear() method?
		if (_tptr) {
			delete _tptr;

			_tptr = nullptr;
		}
		
		if (other._tptr)
			_tptr = other._tptr->copy();
		
		_label = other._label;
		_class = other._class;
		_nodes = other._nodes;
		_leaves = other._leaves;
	}

	return *this;
}

node::~node()
{
	if (_tptr)
		delete _tptr;
	
	_tptr = nullptr;
}

Token *node::copy_token() const
{
	if (_tptr)
		return _tptr->copy();

	return nullptr;
}

node &node::operator[](size_t i)
{
	return _leaves[i];
}

const node &node::operator[](size_t i) const
{
	return _leaves[i];
}

// Write to file
void node::write(std::ostream &os) const
{
	os << "[tptr]";

	size_t len = _leaves.size();
	os.write((char *) (&len), sizeof(size_t));
	for (const node &nd : _leaves)
		nd.write(os);
}

// Position iterators
node::LeavesIt node::begin()
{
	return _leaves.begin();
}

node::LeavesIt node::end()
{
	return _leaves.end();
}

node::LeavesCit node::begin() const
{
	return _leaves.begin();
}

node::LeavesCit node::end() const
{
	return _leaves.end();
}

// Leaves modifiers
node::LeavesIt node::insert(const node::LeavesIt &it, const node &nd)
{
	return _leaves.insert(it, nd);
}

node::LeavesIt node::erase(const node::LeavesIt &it)
{
	return _leaves.erase(it);
}

// Clear leaves
void node::clear()
{
	_leaves.clear();
}

// Properties
bool node::null() const
{
	return (_tptr == nullptr);
}

lbl node::label() const
{
	return _label;
}

bool node::empty() const
{
	return (_tptr == nullptr) && _leaves.empty();
}

Token *node::ptr() const
{
	return _tptr;
}

size_t node::child_count() const
{
	return _leaves.size();
}

Token::type node::caller() const
{
	if (_tptr)
		return _tptr->caller();

	return Token::undefined;
}

// Setters
void node::relabel(lbl label)
{
	_label = label;
}

void node::retokenize(Token *tptr)
{
	// Delete the current token
	if (_tptr)
		delete _tptr;
	
	_tptr = tptr->copy();
}

void node::releaf(const node::Leaves &lvs)
{
	_leaves = lvs;
}

void node::set_count(size_t i)
{
	_nodes = i;
}

void node::transfer(const node &ref)
{
	if (ref._tptr) {
		if (_tptr)
			delete _tptr;
		
		_tptr = ref._tptr->copy();
	}

	_label = ref._label;
	_class = ref._class;
	_nodes = ref._nodes;
	_leaves = ref._leaves;
}

void node::copy_leaves(const node &ref)
{
	_leaves = ref._leaves;
}

void node::append(const node &ref)
{
	_leaves.push_back(ref);
}

// Needs a better name
void node::append_front(const node &ref)
{
	_leaves.insert(_leaves.begin(), ref);
}

void node::remove_end()
{
	if (_leaves.size() > 0)
		_leaves.erase(_leaves.end());
}

void node::print(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";

		counter--;
	}

	if (_tptr) {
		std::cout << "#" << num << ": " << _tptr->dbg_str()
			<< " (" << _tptr << ", " << strlabs[_label]
			<< ") @ " << this << std::endl;	
	} else {
		std::cout << "#" << num << ": null ("
			<< _tptr << ", " << strlabs[_label] << ") @ "
			<< this << std::endl;
	}

	counter = 0;
	for (node itr : _leaves)
		itr.print(++counter, lev + 1);
}

void node::print(std::ostream &os, int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		os << "\t";

		counter--;
	}

	if (_tptr) {
		os << "#" << num << ": " << _tptr->dbg_str()
			<< " (" << _tptr << ", " << strlabs[_label]
			<< ") @ " << this << std::endl;	
	} else {
		os << "#" << num << ": null ("
			<< _tptr << ", " << strlabs[_label] << ") @ "
			<< this << std::endl;
	}

	counter = 0;
	for (node itr : _leaves)
		itr.print(os, ++counter, lev + 1);
}

void node::print_no_address(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	if (_tptr) {
		std::cout << "#" << num << ": " << _tptr->dbg_str() << " (" <<
			strlabs[_label] << ") " << _nodes << " nodes" << ::std::endl;
	} else {
		std::cout << "#" << num << ": null (" <<
			strlabs[_label] << ") " << _nodes << " nodes" << ::std::endl;
	}

	counter = 0;
	for (node itr : _leaves)
		itr.print_no_address(++counter, lev + 1);
}

void node::print_no_address(std::ostream &os, int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		os << "\t";
		counter--;
	}

	if (_tptr) {
		os << "#" << num << ": " << _tptr->dbg_str() << " (" <<
			strlabs[_label] << ") " << _nodes << " nodes" << ::std::endl;
	} else {
		os << "#" << num << ": null (" <<
			strlabs[_label] << ") " << _nodes << " nodes" << ::std::endl;
	}

	counter = 0;
	for (node itr : _leaves)
		itr.print_no_address(os, ++counter, lev + 1);
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

	if (_tptr) {
		oss << "#" << num << ": " << _tptr->dbg_str() << " (" << _tptr
			<< ") @ " << this << " "  << _nodes << " nodes" << std::endl;
	} else {
		oss << "#" << num << ": [Null](" << _tptr
			<< ") @ " << this << " "  << _nodes << " nodes" << std::endl;
	}

	counter = 0;
	for (node itr : _leaves)
		oss << itr.display(++counter, lev + 1);

	return oss.str();
}

// Make non-member
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
void nullify_tree_refs(node &ref, const Args &args)
{
	node_reference *nptr;
	if ((nptr = ref.cast <node_reference> ())) {
		if (std::find(args.begin(), args.end(),
				nptr->symbol()) != args.end())
			nptr->set(nullptr);
	}

	for (node &child : ref)
		nullify_tree_refs(child, args);
}

node factorize(const node &ref, const node &factor)
{
	using namespace std;
	// First check equality
	//
	// TODO: Create a nodecmp function (refactor loose_match)
	if (node::loose_match(ref, factor)) {
		cout << "\tRETURN UNIT" << endl;
		return node(new OpZ(1));
	}
	
	// Proceed normally
	std::vector <node> factors;
	std::stack <node> process;

	process.push(ref);

	node top;
	while (!process.empty()) {
		top = process.top();

		process.pop();

		operation_holder *ophptr = top.cast <operation_holder> ();

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
