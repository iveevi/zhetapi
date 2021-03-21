#include <function.hpp>
#include <engine.hpp>

namespace zhetapi {

// Static variables
Engine Function::engine = Engine();

double Function::h = 0.0001;

// Constructors
Function::Function() : __threads(1) {}

Function::Function(const char *str) : Function(std::string
		(str)) {}

// Symbolic construction
Function::Function(const std::string &str) : __threads(1)
{
	std::string pack;
	std::string tmp;

	size_t count;
	size_t index;
	size_t start;
	size_t end;
	size_t i;

	bool valid;
	bool sb;
	bool eb;

	count = 0;

	valid = false;
	sb = false;
	eb = false;

	// Split string into expression and symbols
	for (i = 0; i < str.length(); i++) {
		if (str[i] == '=') {
			valid = true;
			index = i;

			++count;
		}
	}

	if (!valid || count != 1)
		throw invalid_definition();

	__symbol = str.substr(0, index);


	// Determine parameters' symbols
	for (start = -1, i = 0; i < __symbol.length(); i++) {
		if (str[i] == '(' && start == -1) {
			start = i;
			sb = true;
		}

		if (str[i] == ')') {
			end = i;
			eb = true;
		}
	}

	if (!sb || !eb)
		throw invalid_definition();

	pack = __symbol.substr(start + 1, end - start - 1);

	for (i = 0; i < pack.length(); i++) {
		if (pack[i] == ',' && !tmp.empty()) {
			__params.push_back(tmp);
			
			tmp.clear();
		} else if (!isspace(pack[i])) {
			tmp += pack[i];
		}
	}

	if (!tmp.empty())
		__params.push_back(tmp);
	
	// Determine function's symbol
	__symbol = __symbol.substr(0, start);

	// Construct the tree manager
	__manager = node_manager(str.substr(++index), __params, &engine);

	__manager.simplify();
}

// Symbolic constructor with external symbol table
Function::Function(const std::string &str, Engine *bptr) : __threads(1)
{
	// TODO: Remove this (duplication)
	std::string pack;
	std::string tmp;

	size_t count;
	size_t index;
	size_t start;
	size_t end;
	size_t i;

	bool valid;
	bool sb;
	bool eb;

	count = 0;

	valid = false;
	sb = false;
	eb = false;

	// Split string into expression and symbols
	for (i = 0; i < str.length(); i++) {
		if (str[i] == '=') {
			valid = true;
			index = i;

			++count;
		}
	}

	if (!valid || count != 1)
		throw invalid_definition();

	__symbol = str.substr(0, index);


	// Determine parameters' symbols
	for (start = -1, i = 0; i < __symbol.length(); i++) {
		if (str[i] == '(' && start == -1) {
			start = i;
			sb = true;
		}

		if (str[i] == ')') {
			end = i;
			eb = true;
		}
	}

	if (!sb || !eb)
		throw invalid_definition();

	pack = __symbol.substr(start + 1, end - start - 1);

	for (i = 0; i < pack.length(); i++) {
		if (pack[i] == ',' && !tmp.empty()) {
			__params.push_back(tmp);
			
			tmp.clear();
		} else if (!isspace(pack[i])) {
			tmp += pack[i];
		}
	}

	if (!tmp.empty())
		__params.push_back(tmp);
	
	// Determine function's symbol
	__symbol = __symbol.substr(0, start);

	// Construct the tree manager
	__manager = node_manager(str.substr(++index), __params, bptr);

	__manager.simplify();
}

// Member-wise construction
Function::Function(const std::string &symbol, const std::vector
		<std::string> &params, const node_manager &manager) :
		__symbol(symbol), __params(params),
		__manager(manager), __threads(1) {}

Function::Function(const Function &other) :
		__symbol(other.__symbol), __params(other.__params),
		__manager(other.__manager), __threads(1) {}

// Getters
bool Function::is_variable(const std::string &str) const
{
	return std::find(__params.begin(), __params.end(), str)
		!= __params.end();
}

std::string &Function::symbol()
{
	return __symbol;
}

const std::string Function::symbol() const
{
	return __symbol;
}

void Function::set_threads(size_t threads)
{
	__threads = threads;
}

// Computational utilities
Token *Function::operator()(std::vector <Token *> toks)
{
	assert(toks.size() == __params.size());

	return __manager.substitute_and_compute(toks, __threads);
}

template <class ... A>
Token *Function::derivative(const std::string &str, A ... args)
{
	std::vector <Token *> Tokens;

	gather(Tokens, args...);

	assert(Tokens.size() == __params.size());

	size_t i = index(str);

	assert(i != -1);

	// Right
	Token *right;

	Tokens[i] = engine.compute("+", {Tokens[i], new Operand <double> (h)});

	for (size_t k = 0; k < Tokens.size(); k++) {
		if (k != i)
			Tokens[k] = Tokens[k]->copy();
	}
	
	right = __manager.substitute_and_compute(Tokens);
	
	// Left
	Token *left;

	Tokens[i] = engine.compute("-", {Tokens[i], new Operand <double> (2.0 * h)});

	for (size_t k = 0; k < Tokens.size(); k++) {
		if (k != i)
			Tokens[k] = Tokens[k]->copy();
	}

	left = __manager.substitute_and_compute(Tokens);

	// Compute
	Token *diff = engine.compute("-", {right, left});

	diff = engine.compute("/", {diff, new Operand <double> (2.0 * h)});

	return diff;
}

Function Function::differentiate(const std::string &str) const
{
	// Improve naming later
	std::string name = "d" + __symbol + "/d" + str;

	node_manager dm = __manager;

	dm.differentiate(str);

	Function df = Function(name, __params, dm);

	return df;
}

// Virtual functions
Token::type Function::caller() const
{
	return Token::ftn;
}

std::string Function::str() const
{
	return display();
}

Token *Function::copy() const
{
	return new Function(*this);
}

bool Function::operator==(Token *tptr) const
{
	Function *ftn = dynamic_cast <Function *> (tptr);

	if (!ftn)
		return false;
	
	return ftn->__symbol == __symbol;
}

// Printing utilities
void Function::print() const
{
	__manager.print();
}

std::string Function::display() const
{
	std::string str = __symbol + "(";

	size_t n = __params.size();
	for (size_t i = 0; i < n; i++) {
		str += __params[i];
		
		if (i < n - 1)
			str += ", ";
	}

	str += ") = " + __manager.display();

	return str;
}

std::ostream &operator<<(std::ostream &os, const Function &ftr)
{
	os << ftr.display();
	return os;
}

// Comparing
bool operator<(const Function &a, const Function &b)
{
	return a.symbol() < b.symbol();
}

bool operator>(const Function &a, const Function &b)
{
	return a.symbol() > b.symbol();
}

size_t Function::index(const std::string &str) const
{
	auto itr = std::find(__params.begin(), __params.end(), str);

	if (itr == __params.end())
		return -1;

	return std::distance(__params.begin(), itr);
}

}
