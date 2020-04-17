#ifndef OPERATION_H
#define OPERATION_H

// C++ Standard Libraries
#include <vector>
#include <iostream>
#include <sstream>

// Custom Built Libraries
#include "token.h"
#include "exception.h"
#include "operand.h"

/* Operation Class:
 *
 * Represents an operation in mathematics
 * that bases computations on the oper_t
 * operand class, which can check whether
 * the operand class supports the computations
 * and makes sure that the class passed in
 * is an operand class or its derived class */
template <class oper_t>
class operation : public token {
public:
	/* Argset_Exception Class:
	 *
	 * An error class derived from
	 * operation::exception that represents
	 * an error resulting from receiving an
	 * inappropriate number of operands
	 * in the compute method */
	class argset_exception : public exception {
	public:
		/* Constructors:
		 * argset_exception() - default constructor
		 *   that sets the private member variable
		 *   [msg] to an empty string 
		 * argset_exception(std::string) - constructor
		 *   that sets the private member variable [msg]
		 *   to whatever string is passed 
		 * argset_exception(std::string, int,
		 *   int) - constructor that creates a message
		 *   using the name of the operation, number of
		 *   operands required and the number of operands
		 *   received
		 * argset_exception(int, const operation
		 *   <oper_t> &) - constructor that creates
		 *   the message using the number of operands
		 *   (passed) and the operation it is called
		 *   from (passed) */
		argset_exception();
		explicit argset_exception(std::string);
		argset_exception(std::string, int, int);
		argset_exception(int, const operation <oper_t> &);

		/* Virtualized Member Functions:
		 * void [set](std::string, int, int) - setter
		 *   method that does the same thing as
		 *   the corresponding constructor
		 * void [set](int, const operation <oper_t> &) - setter
		 *   method that does the same thing as
		 *   the corresponding constructor */
		virtual void set(std::string, int, int);
		virtual void set(int, const operation <oper_t> &);
	};

	/* Computation_Exception Class:
	 *
	 * Represents an error derived from trying
	 * to compute something using the operand
	 * class */
	class computation_exception : public exception {};

	/* Illegal_Type_Exception Class:
	 *
	 * Represents an error derived from trying
	 * to create an operation object by passing
	 * a non-operand class as a parameter */
	class illegal_type_exception : public exception {};
	
	/* Typedef oper_t (*[function])(const std::vector
	 * <oper_t> &) - a type alias for convenience for
	 * the type of function used for computation */
	typedef oper_t (*function)(const std::vector <oper_t> &);

	/* Enum orders - the type parameters
	 * for pemdas values of operations
	 * NA_L0 - for empty operations
	 * SA_L1 - for addition/subtraction
	 * MDM_L2 - for multiplication/division,
	 *   and modulus
	 * FUNC_LMAX - for function std operations
	 *   such as sin, cos, log, etc. */
	enum order {SA_L1, MDM_L2, EXP_L3, FUNC_LMAX, NA_L0};

	/* Constructors:
	 * operation() - sets the private member function
	 *   [func] to a null pointer, [opers] to -1,
	 *   [symbols] and [name] to null pointer as well
	 * operation(std::string, function, int, const
	 *   std::vector> &, order) - sets the private
	 *   member variables to the corresponding parameters
	 *   passed as parameters */
	operation() noexcept(false);
	operation(const operation &) noexcept(false);
	operation(std::string, function, int, const std::vector
		<std::string> &, order, const std::vector <std::string> &)
		noexcept(false);
	~operation();

	/* Virtualized Member Functions:
	 * void [set](std::string, function, int, const std::vector
	 *   <std::string> &) - sets the private member variables to
	 *   the corresponding parameters passed
	 * void [function][get] const - returns a pointer to the
	 *   computer function used by the object
	 * void [function]operator*() const - returns a pointer to
	 *   the computer function used by the object
	 * void oper_t [compute](const std::vector <oper_t> &) - returns
	 *   the result of the operation given a list of operands, and
	 *   throws [argset_exception] if there are not exactly [opers]
	 *   operands
	 * void oper_t operator()(const std::vector <oper_t> &) - does
	 *   the same thing as compute but is supported by the function
	 *   operator */
	virtual void set(std::string, function, int, const std::vector
		<std::string> &, order, const std::vector <std::string> &);
	virtual void operator()(std::string, function, int, const std::vector
		<std::string> &, order, const std::vector <std::string> &);

	virtual function get() const;
	virtual function operator*() const;

	virtual oper_t compute(const std::vector <oper_t> &) const
		noexcept(false);
	virtual oper_t operator()(const std::vector <oper_t> &) const
		noexcept(false);

	virtual bool matches(const std::string &) const;
	virtual bool operator[](const std::string &) const;
	
	virtual order get_order() const;
	virtual size_t operands() const {return opers;}

	operation &operator=(const operation &);
	
	type caller() const override;
	std::string str() const override;

	const std::string &symbol() const
	{
		return symbols[0];
	}
	
	/* Friend Functions:
	 * std::ostream &operator<<(std::ostream &, const operation
	 *   <oper_t> &) - outputs information about the passed
	 *   operation
	 *   object into the stream pointed to by the passed ostream */
	template <class T>
	friend std::ostream &operator<<(std::ostream &, const operation
		<T> &);

	/* Comparison Functions:
	 *
	 * The following comparison operators
	 * execute by comparing the pemdas values
	 * except in == and != (comparison of all
	 * members is done) */
	template <class T>
	friend bool operator==(const operation <T> &, const operation
		<T> &);
	
	template <class T>
	friend bool operator!=(const operation <T> &, const operation
		<T> &);
	
	template <class T>
	friend bool operator>(const operation <T> &, const operation
		<T> &);
	
	template <class T>
	friend bool operator<(const operation <T> &, const operation
		<T> &);
	
	template <class T>
	friend bool operator>=(const operation <T> &, const operation
		<T> &);
	
	template <class T>
	friend bool operator<=(const operation <T> &, const operation
		<T> &);
private:
	/* [function] [func] - the function used by the operation */
	function func;

	/* std::string [name] - the name of the operation */
	std::string name;

	/* std::size_t [opers] - the number of operands expected
	 * by the operation's function */
	std::size_t opers;

	/* std::vector <std::string> [symbols] - the list of
	 * symbols of the operation that are used in parsing */
	std::vector <std::string> symbols;

	/* [order] [pemdas] - the operations pemdas value, see
	 * public declaration for a description of all the types
	 * of pemdas orders/values */
	order pemdas;

	/* std::vector <std::string> [formats] - the list of string
	 * formats of the operation in the user input that are
	 * or can be valid */
	std::vector <std::string> formats;

	/* bool [is_same](std::string, std::string) - checks if two
	 * strings are the same with the addition of wildcards */
	bool is_same(const std::string &, const std::string &);
};

/* Default operation specification using
 * the default operand type/class */
typedef operation <num_t> opn_t;

/* Operation Class Member Functions
 *
 * See declaration to see a
 * description of each function
 *
 * Constructors: */
template <class oper_t>
operation <oper_t> ::operation() : name(""), opers(0), symbols(0),
	func(nullptr), pemdas(NA_L0), formats(0)
{
	/* find out later if (typeid(oper_t) == operand) {
		throw new illegal_type_exception("Expected a"
			    "template operand class");
	} */
}

template <class oper_t>
operation <oper_t> ::operation(const operation &other) : name(other.name),
	opers(other.opers), symbols(other.symbols), func(other.func),
	pemdas(other.pemdas), formats(other.formats)
{
	/* if (!std::is_same <operand, oper_t> ::value) {
		throw new illegal_type_exception("Expected a"
			    "template operand class");
	} */
}

template <class oper_t>
operation <oper_t> ::operation(std::string str, function nfunc,
	int nopers, const std::vector <std::string> &nsymbols, order pm,
	const std::vector <std::string> &nformats) : name(str), opers(nopers),
	symbols(nsymbols), func(nfunc), pemdas(pm), formats(nformats)
{
	/* if (!std::is_same <operand, oper_t> ::value) {
		throw new illegal_type_exception("Expected a"
			    "template operand class");
	} */
}

template <class oper_t>
operation <oper_t> ::~operation()
{
	// Put the operation to a stack
	// so that future uses can be reused
	// instead remade
	// delete func; <- illegal
}

/* Private member functions:
 * helper functions that are put
 * outside the scope of the general
 * program */
template <class oper_t>
bool operation <oper_t> ::is_same(const std::string &left, const
		std::string &right)
{
	std::cout << "Unimplemeted XP" << std::endl;
	return false;
}

/* Virtualized Member Functions:
 * setters, getters, and operators */
template <class oper_t>
void operation <oper_t> ::set(std::string str, function nfunc,
		int nopers, const std::vector <std::string> &nsymbols,
		order pm, const std::vector <std::string> &nformats)
{
	name = str;
	func = nfunc;
	opers = nopers;
	symbols = nsymbols;
	pemdas = pm;
	formats = nformats;
}

template <class oper_t>
void operation <oper_t> ::operator()(std::string str, function nfunc,
		int nopers, const std::vector <std::string> &nsymbols,
		order pm, const std::vector <std::string> &nformats)
{
	name = str;
	func = nfunc;
	opers = nopers;
	symbols = nsymbols;
	pemdas = pm;
	formats = nformats;
}

template <class oper_t>
typename operation <oper_t>::function operation <oper_t> ::get()
		const
{
	return func;
}

template <class oper_t>
typename operation <oper_t>::function operation <oper_t> ::operator*()
		const
{
	return func;
}

template <class oper_t>
oper_t operation <oper_t> ::compute(const std::vector <oper_t> &inputs)
		const
{
	if (inputs.size() != opers)
		throw argset_exception(inputs.size(), *this);
	return (*func)(inputs);
}

template <class oper_t>
oper_t operation <oper_t> ::operator()(const std::vector <oper_t> &inputs)
		const
{
	if (inputs.size() != opers)
		throw argset_exception(inputs.size(), *this);
	return (*func)(inputs);
}

template <class oper_t>
typename operation <oper_t> ::order operation <oper_t> ::get_order()
		const
{
	return pemdas;
}

template <class oper_t>
operation <oper_t> &operation <oper_t> ::operator=(const operation <oper_t>
		&other)
{
	name = other.name;-
	opers = other.opers;
	symbols = other.symbols;
	func = std::move(other.func);
}

template <class oper_t>
bool operation <oper_t> ::matches(const std::string &str) const
{
	for (std::string s : symbols) {
		if (s == str)
			return true;
	}

	return false;
}


template <class oper_t>
bool operation <oper_t> ::operator[](const std::string &str) const
{
	for (std::string s : symbols) {
		if (s == str)
			return true;
	}

	return false;
}

/* Friend Functions: ostream utilities */
template <class oper_t>
std::ostream &operator<<(std::ostream &os, const operation <oper_t>
		&opn)
{
	os << opn.name << " - " << opn.opers << " operands, ";
	for (std::string str : opn.symbols)
		os << str << " ";
	return os;
}

/* Comparison functions: order described in
 * class declaration */
template <class oper_t>
bool operator==(const operation <oper_t> &right, const operation <oper_t>
		&left)
{
	return right.name == left.name;
	/*if (right.pemdas != left.pemdas)
		return false;
	if (right.opers != left.opers)
		return false;
	if (right.name != left.name)
		return false;
	if (right.symbols != left.symbols)
		return false;
	return right.func == left.func;*/
}

template <class oper_t>
bool operator!=(const operation <oper_t> &right, const operation <oper_t>
		&left)
{
	if (right.pemdas != left.pemdas)
		return true;
	if (right.opers != left.opers)
		return true;
	if (right.name != left.name)
		return true;
	if (right.symbols != left.symbols)
		return true;
	return right.func != left.func;
}

template <class oper_t>
bool operator>(const operation <oper_t> &right, const operation <oper_t>
		&left)
{
	return right.pemdas > left.pemdas;
}

template <class oper_t>
bool operator<(const operation <oper_t> &right, const operation <oper_t>
		&left)
{
	return right.pemdas < left.pemdas;
}

template <class oper_t>
bool operator>=(const operation <oper_t> &right, const operation <oper_t>
		&left)
{
	return (right > left) || (right == left);
}

template <class oper_t>
bool operator<=(const operation <oper_t> &right, const operation <oper_t>
		&left)
{
	return (right < left) || (right == left);
}

/* Token class derived functions: */
template <class oper_t>
token::type operation <oper_t> ::caller() const
{
	return OPERATION;
}

template <class oper_t>
std::string operation <oper_t> ::str() const
{
	std::ostringstream scin;
	scin << name << " - " << opers << " operands, ";
	for (std::string str : symbols)
		scin << str << " ";
	return scin.str();
}

/* Argset Exception Class Member Functions
 *
 * See class declaration for a description
 * of each member function
 *
 * Constructors */
template <class oper_t>
operation <oper_t> ::argset_exception::argset_exception()
	: exception() {}

template <class oper_t>
operation <oper_t> ::argset_exception::argset_exception(int actual,
		const operation <oper_t> &obj)
{
	using std::to_string;
	exception::msg = obj.name + ": Expected " + to_string(obj.opers);
	exception::msg += " operands, received " + to_string(actual);
	exception::msg += " instead.";
}

template <class oper_t>
operation <oper_t> ::argset_exception::argset_exception(std::string str)
		: exception(str) {}

template <class oper_t>
operation <oper_t> ::argset_exception::argset_exception(std::string str,
		int expected, int actual)
{
	using std::to_string;
	exception::msg = str + ": Expected " + to_string(expected);
	exception::msg += " operands, received " + to_string(actual);
	exception::msg += "instead.";
}

/* Virtualized member functions:
 * setters only - getters are inherited */
template <class oper_t>
void operation <oper_t> ::argset_exception::set(int actual, const
		operation <oper_t> &obj)
{
	using std::to_string;
	exception::msg = obj.name + ": Expected " + to_string(obj.opers);
	exception::msg += " operands, received " + to_string(actual) + " instead.";
}

template <class oper_t>
void operation <oper_t> ::argset_exception::set(std::string str,
		int expected, int actual)
{
	using std::to_string;
	exception::msg = str + ": Expected " + to_string(expected);
	exception::msg += " operands, received " + to_string(actual) + "instead.";
}

#endif
