#ifndef FUNCTOR_H_
#define FUNCTOR_H_

// C/C++ headers
#include <ostream>
#include <string>

#include <dlfcn.h>

// Engine headers
#include <core/node_manager.hpp>

namespace zhetapi {

/*
 * Represents a mathematical function.
 */
template <class T, class U>
class Function : public Token {
	::std::string			__symbol;
	::std::vector <::std::string>	__params;
	node_manager <T, U>		__manager;
	size_t				__threads;
public:
	Function();
	Function(const char *);
	Function(const ::std::string &);

	Function(const ::std::string &, const ::std::vector <::std::string>
			&, const node_manager <T, U> &);

	Function(const Function &);

	::std::string &symbol();
	const ::std::string symbol() const;

	void set_threads(size_t);

	Token *operator()(::std::vector <Token *>);

	template <class ... A>
	Token *operator()(A ...);

	template <size_t, class ... A>
	Token *operator()(A ...);

	template <class ... A>
	Token *derivative(const ::std::string &, A ...);

	Function <T, U> differentiate(const ::std::string &) const;

	template <class A, class B>
	friend bool operator<(const Function <A, B> &, const Function <A, B> &);

	template <class A, class B>
	friend bool operator>(const Function <A, B> &, const Function <A, B> &);

	::std::string generate_general() const;

	void *compile_general() const;

	// Virtual overloads
	Token::type caller() const override;
	::std::string str() const override;
	Token *copy() const override;
	bool operator==(Token *) const override;

	// Printing
	void print() const;

	::std::string display() const;

	template <class A, class B>
	friend ::std::ostream &operator<<(::std::ostream &, const Function <A, B> &);
private:
	template <class A>
	void gather(::std::vector <Token *> &, A);

	template <class A, class ... B>
	void gather(::std::vector <Token *> &, A, B ...);

	size_t index(const ::std::string &) const;
public:
	// Exception classes
	class invalid_definition {};

	// Static variables
	static Barn <T, U> barn;

	static T h;
};

// Static
template <class T, class U>
Barn <T, U> Function <T, U> ::barn = Barn <T, U> ();

template <class T, class U>
T Function <T, U> ::h = 0.0001;

// Constructors
template <class T, class U>
Function <T, U> ::Function() : __threads(1) {}

template <class T, class U>
Function <T, U> ::Function(const char *str) : Function(::std::string
		(str)) {}

template <class T, class U>
Function <T, U> ::Function(const ::std::string &str) : __threads(1)
{
	::std::string pack;
	::std::string tmp;

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
	__manager = node_manager <T, U> (str.substr(++index), __params, barn);

	__manager.simplify();
}

// Member-wise construction
template <class T, class U>
Function <T, U> ::Function(const ::std::string &symbol, const ::std::vector
		<::std::string> &params, const node_manager <T, U>
		&manager) : __symbol(symbol), __params(params),
		__manager(manager), __threads(1) {}

template <class T, class U>
Function <T, U> ::Function(const Function <T, U> &other) :
	__symbol(other.__symbol), __params(other.__params),
	__manager(other.__manager), __threads(1) {}

// Getters
template <class T, class U>
::std::string &Function <T, U> ::symbol()
{
	return __symbol;
}

template <class T, class U>
const ::std::string Function <T, U> ::symbol() const
{
	return __symbol;
}

template <class T, class U>
void Function <T, U> ::set_threads(size_t threads)
{
	__threads = threads;
}

// Computational utilities
template <class T, class U>
Token *Function <T, U> ::operator()(::std::vector <Token *> toks)
{
	assert(toks.size() == __params.size());

	return __manager.substitute_and_compute(toks, __threads);
}

template <class T, class U>
template <class ... A>
Token *Function <T, U> ::operator()(A ... args)
{
	::std::vector <Token *> Tokens;

	gather(Tokens, args...);

	assert(Tokens.size() == __params.size());

	return __manager.substitute_and_compute(Tokens, __threads);
}

template <class T, class U>
template <class ... A>
Token *Function <T, U> ::derivative(const ::std::string &str, A ... args)
{
	::std::vector <Token *> Tokens;

	gather(Tokens, args...);

	assert(Tokens.size() == __params.size());

	size_t i = index(str);

	assert(i != -1);

	// Right
	Token *right;

	Tokens[i] = barn.compute("+", {Tokens[i], new Operand <T> (h)});

	for (size_t k = 0; k < Tokens.size(); k++) {
		if (k != i)
			Tokens[k] = Tokens[k]->copy();
	}
	
	right = __manager.substitute_and_compute(Tokens);
	
	// Left
	Token *left;

	Tokens[i] = barn.compute("-", {Tokens[i], new Operand <T> (T(2) * h)});

	for (size_t k = 0; k < Tokens.size(); k++) {
		if (k != i)
			Tokens[k] = Tokens[k]->copy();
	}

	left = __manager.substitute_and_compute(Tokens);

	// Compute
	Token *diff = barn.compute("-", {right, left});

	diff = barn.compute("/", {diff, new Operand <T> (T(2) * h)});

	return diff;
}

template <class T, class U>
Function <T, U> Function <T, U> ::differentiate(const ::std::string &str) const
{
	// Improve naming later
	::std::string name = "d" + __symbol + "/d" + str;

	node_manager <T, U> dm = __manager;

	dm.differentiate(str);

	Function <T, U> df = Function <T, U> (name, __params, dm);

	return df;
}

template <class T, class U>
::std::string Function <T, U> ::generate_general() const
{
	using namespace std;

	// cout << endl << "Generating" << endl;
	


	::std::string file;

	file = "__gen_" + __symbol;

	// cout << "\tname: " << file << endl;

	__manager.generate(file);

	return file;
}

template <class T, class U>
void *Function <T, U> ::compile_general() const
{
	::std::string file = generate_general();

#ifdef __linux__

#define ZHP_FUNCTION_COMPILE_GENERAL

	int ret;

	ret = system("mkdir -p gen");

	// Add testing flags for compiling with "-g"
	ret = system(("g++-8 --no-gnu-unique -I engine -I inc/hidden -I inc/std \
				-g -rdynamic -fPIC -shared " + file  +
				".cpp -o gen/" + file +
				".so").c_str());
	
	ret = system(("rm -rf " + file + ".cpp").c_str());

	void *handle = dlopen(("./gen/" + file + ".so").c_str(), RTLD_NOW);

	// ::std::cout << "handle @ " << file << ": " << handle << ::std::endl;

	const char *dlsym_error = dlerror();

	if (dlsym_error) {
		::std::cerr << "Cannot load symbol '" << file << "': " << dlsym_error << '\n';
		
		// dlclose(handle);
		
		return nullptr;
	}

	void *ptr = dlsym(handle, file.c_str());

	// ::std::cout << "\tptr: " << ptr << ::std::endl;
	// ::std::cout << "\tfile: " << file << ::std::endl;

	// dlsym_error = dlerror();

	if (dlsym_error) {
		::std::cerr << "Cannot load symbol '" << file << "': " << dlsym_error << '\n';
		
		// dlclose(handle);
		
		return nullptr;
	}

	// dlclose(handle);

	dlsym_error = dlerror();

	if (dlsym_error) {
		::std::cerr << "Cannot close for symbol '" << file << "': " << dlsym_error << '\n';
		
		// dlclose(handle);
		
		return nullptr;
	}

	return ptr;
#else

#warning "No support for Function::compile_general in the current operating system"

	return nullptr;
#endif

}

// Virtual functions
template <class T, class U>
Token::type Function <T, U> ::caller() const
{
	return Token::ftn;
}

template <class T, class U>
::std::string Function <T, U> ::str() const
{
	return display();
}

template <class T, class U>
Token *Function <T, U> ::copy() const
{
	return new Function <T, U> (*this);
}

template <class T, class U>
bool Function <T, U> ::operator==(Token *tptr) const
{
	Function <T, U> *ftn = dynamic_cast <Function <T, U> *> (tptr);

	if (!ftn)
		return false;
	
	return ftn->__symbol == __symbol;
}

// Printing utilities
template <class T, class U>
void Function <T, U> ::print() const
{
	__manager.print();
}

template <class T, class U>
::std::string Function <T, U> ::display() const
{
	::std::string str = __symbol + "(";

	size_t n = __params.size();
	for (size_t i = 0; i < n; i++) {
		str += __params[i];
		
		if (i < n - 1)
			str += ", ";
	}

	str += ") = " + __manager.display();

	return str;
}

template <class T, class U>
::std::ostream &operator<<(::std::ostream &os, const Function <T, U> &ftr)
{
	os << ftr.display();
	return os;
}

// Comparing
template <class T, class U>
bool operator<(const Function <T, U> &a, const Function <T, U> &b)
{
	return a.symbol() < b.symbol();
}

template <class T, class U>
bool operator>(const Function <T, U> &a, const Function <T, U> &b)
{
	return a.symbol() > b.symbol();
}

// Gathering facilities
template <class T, class U>
template <class A>
void Function <T, U> ::gather(::std::vector <Token *> &Tokens, A in)
{
	Tokens.push_back(new Operand <A>(in));
}

template <class T, class U>
template <class A, class ... B>
void Function <T, U> ::gather(::std::vector <Token *> &Tokens, A in, B ... args)
{
	Tokens.push_back(new Operand <A>(in));

	gather(Tokens, args...);
}

template <class T, class U>
size_t Function <T, U> ::index(const ::std::string &str) const
{
	auto itr = ::std::find(__params.begin(), __params.end(), str);

	if (itr == __params.end())
		return -1;

	return ::std::distance(__params.begin(), itr);
}

// External classes
template <class T, class U>
void Barn <T, U> ::put(Function <T, U> ftr)
{
	if (fstack.contains(ftr.symbol()))
		fstack.get(ftr.symbol()) = ftr;
	else
		fstack.insert(ftr);
}

template <class T, class U>
Function <T, U> &Barn <T, U> ::retrieve_function(const ::std::string &str)
{
	return fstack.get(str);
}

}

#endif
