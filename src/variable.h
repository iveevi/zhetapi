#ifndef VARIALBE_H
#define VARIALBE_H

// C++ Standard Libraries
#include <string>
#include <iostream>
#include <sstream>

// Custom Built Libraries
#include "token.h"
#include "exception.h"

/* Variable Class:
 *
 * Represents a variable in mathematics
 * which can be used as a dummy variable
 * for custom functions and has the safe
 * ability to read and write the value
 * is stores */
template <typename data_t>
class variable : public token {
	/* std::string [name] - the name of the
	 * variable that will be taken into account
	 * when creating custom functions and using
	 * them later on */
	std::string name;

	/* data_t [val] - the value stored by the
	 * object which serves the same purpose as
	 * the val member in the operand class */
	data_t val;

	/* bool [param] - a boolean variable that
	 * represents whether or not the current
	 * variable object is being used a parameter
	 * (dummy) or to hold information */
	bool param;
public:
	/*
	 * Bypass_Attempt_Exception Class:
	 *
	 * Represents the error thrown whenever
	 * the value of the variable object is
	 * attempted to be accessed while it is
	 * supposed to be a parameter
	 */

	// move later
	class bypass_attempt_exception {};

	/* Dummy parameters for operators () and *
	 * one for dummy int and one for dummy
	 * std::string */
	#define DINT 0x0
	#define DSTRING ""

	/* Constructors:
	 * variable() - creates a new variable object
	 *   that is by default a dummy parameter
	 * variable(std::string, boolean, data_t) - creates
	 *   a new variable object with corresponding
	 *   members with respect to the passed parameters,
	 *   private member variable [val] is defaulted
	 *   to the default value of data_t */
	variable();
	variable(const std::string &, const data_t & = data_t());

	// change to const refs
	variable(std::string, bool, data_t = data_t());

	/* Virtualized Member Functions:
	 * void [set](bool) - changes whether or not the
	 *   current variable object acts as a dummy
	 *   or as a value
	 * void [set](data_t) - sets the private member
	 *   variable [val] to the value that is passed
	 * void [set](std::string) - sets the private
	 *   member variable [name] to the string
	 *   that is passed
	 * void [set](data_t, std::string) - sets
	 *   the name, [name], and value, [val], of
	 *   the current variable object
	 * void operator[](bool) - does the same thing
	 *   as void [set](bool)
	 * void operator[](data_t) - does the same thing
	 *   as void [set](data_t)
	 * void operator[](std::string) - does the same
	 *   thing as void [set](std::string)
	 * void operator[](data_t, std::string) - does the
	 *   same thing as void [set](data_t, std::string)
	 * data_t &[get]() - returns a reference to the
	 *   value of the object or throws a bypass attempt
	 *   exception
	 * const data_t &[get]() const - returns a constant
	 *   (unchangeable) reference to the value of the
	 *   object or throws a bypass attempt exception
	 * const std::string &[get]() const - returns the name
	 *   of the current variable object
	 * const std::pair <data_t, std::string> &[get]()
	 *   const - returns a tuple of the value and name of
	 *   the current variable object, or throws a bypass
	 *   attempt exception if the object is a dummy
	 * data_t &operator() - does the same thing as
	 *   data_t &[get]()
	 * const data_t &operator() const- does the same
	 *   thing as const data_t &[get](); 
	 * const std::string &operator() const - does the
	 *   same thing as const std::string &[get]()
	 * const std::pair <data_t, std::string> &operator()
	 *   const - does the same thing as const std::pair
	 *   <data_t, std::string> &[get]() const */
	virtual void set(bool);
	virtual void set(data_t);
	virtual void set(std::string);
	virtual void set(data_t, std::string);
	
	virtual void operator[](bool);
	virtual void operator[](data_t) noexcept(false);
	virtual void operator[](std::string);

	virtual data_t &get() noexcept(false);
	virtual const data_t &get() const noexcept(false);
	virtual const std::string &get(int) const;
	virtual const std::pair <data_t, std::string>
		&get(std::string) const noexcept(false);

	virtual data_t &operator*() noexcept(false);
	virtual const data_t &operator*() const noexcept(false);
	virtual const std::string &operator*(int) const noexcept(false);
	virtual const std::pair <data_t, std::string>
		&operator*(std::string) const noexcept(false);

	// returns the name of the variable
	virtual const std::string &symbol() const;

	type caller() const override;
	std::string str() const override;
	
	/* Friends:
	 * std::ostream &operator<<(std::ostream &, const variable
	 *   <data_t> &) - outputs the name of the passed variable
	 *   object, its value, and whether or not it is a dummy
	 * std::istream &operator>>(std::istream &, variable &) - reads
	 *   information from the passed istream object, and modifies
	 *   the passed variable object appropriately */
	template <typename type>
	friend std::ostream &operator<<(std::ostream &os, const
		variable <type> &);

	template <typename type>
	friend std::istream &operator>>(std::istream &is, variable
		<type> &);

	/* Comparison Functions:
	 *
	 * The following comparison operators
	 * execute by comparing the values of
	 * the variables, and throw a bypass
	 * attempt error if any of the variables
	 * are parameters */
	template <typename type>
	friend bool operator==(const variable <type> &, const
		variable <type> &) noexcept(false);

	template <typename type>
	friend bool operator!=(const variable <type> &, const
		variable <type> &) noexcept(false);

	template <typename type>
	friend bool operator>(const variable <type> &, const
		variable <type> &) noexcept(false);

	template <typename type>
	friend bool operator<(const variable <type> &, const
		variable <type> &) noexcept(false);

	template <typename type>
	friend bool operator>=(const variable <type> &, const
		variable <type> &) noexcept(false);

	template <typename type>
	friend bool operator<=(const variable <type> &, const
		variable <type> &) noexcept(false);
};

/* Default variable specification using
 * the default numerical type/info structure */
typedef variable <def_t> var_t;

/* Variable Class Member Functions
 *
 * See class declaration for a
 * description of each function
 *
 * Constructors: */
template <typename data_t>
variable <data_t> ::variable() : val(data_t()), name("x"),
	param(false) {}

template <class T>
variable <T> ::variable(const std::string &str, const T &dt)
	: val(dt), name(str), param(false) {}

template <typename data_t>
variable <data_t> ::variable(std::string str, bool bl, data_t vl)
	: val(vl), name(str), param(bl) {}

/* Virtualized member functions:
 * setters, getters and operators */
template <typename data_t>
void variable <data_t> ::set(bool bl)
{
	param = bl;
}

template <typename data_t>
void variable <data_t> ::set(data_t vl)
{
	val = vl;
}

template <typename data_t>
void variable <data_t> ::set(std::string str)
{
	name = str;
}

template <typename data_t>
void variable <data_t> ::set(data_t vl, std::string str)
{
	val = vl;
	name = str;
}

template <typename data_t>
void variable <data_t> ::operator[](bool bl)
{
	param = bl;
}

template <typename data_t>
void variable <data_t> ::operator[](data_t vl)
{
	if (param)
		throw bypass_attempt_exception();
	val = vl;
}

template <typename data_t>
data_t &variable <data_t> ::get()
{
	if (param)
		throw bypass_attempt_exception();
	return val;
}

template <typename data_t>
const data_t &variable <data_t> ::get() const
{
	if (param)
		throw bypass_attempt_exception();
	return val;
}

template <typename data_t>
const std::string &variable <data_t> ::get(int dm) const
{
	return name;
}

template <typename data_t>
const std::pair <data_t, std::string> &variable <data_t> ::get
		(std::string dstr) const
{
	if (param)
		throw bypass_attempt_exception();
	return std::pair <data_t, std::string> (val, name);
}

template <typename data_t>
void variable <data_t> ::operator[](std::string str)
{
	name = str;
}

template <typename data_t>
const data_t &variable <data_t> ::operator*() const
{
	if (param)
		throw bypass_attempt_exception();
	return val;
}

template <typename data_t>
const std::string &variable <data_t> ::operator*(int dm) const
{
	return name;
}

template <typename data_t>
data_t &variable <data_t> ::operator*()
{
	if (param)
		throw bypass_attempt_exception();
	return val;
}

template <typename data_t>
const std::pair <data_t, std::string> &variable <data_t> ::operator*
		(std::string dstr) const
{
	if (param)
		throw bypass_attempt_exception();
	return std::pair <data_t, std::string> (val, name);
}

template <typename data_t>
const std::string &variable <data_t> ::symbol() const
{
	return name;
}

/* Friend functions: ostream and
 * istream utilities */
template <typename data_t>
std::ostream &operator<<(std::ostream &os, const variable <data_t> &var)
{
	os << "[" << var.name << "] - ";

	if (var.param)
		os << " NULL (PARAMETER)";
	else
		os << var.val;

	return os;
}

template <typename data_t>
std::istream &operator>>(std::istream &is, variable <data_t> &var)
{
	// Implement in the general scope
	// Later, after trees
	return is;
}

/* Comparison functions: execute comparison
 * functions by comparing values, and throws
 * a bypass error if one argument is a parameter */
template <typename data_t>
bool operator==(const variable <data_t> &right, const variable <data_t> &left)
{
	// Later, add constructor for
	// Bypass exception that takes in
	// The name of the violated variable
	
	if (right.param || left.param) // Distinguish later
		throw variable <data_t> ::bypass_attempt_exception("needs change");
	return right.name == left.name;
}

template <typename data_t>
bool operator!=(const variable <data_t> &right, const variable <data_t> &left)
{
	if (right.param || left.param)
		throw variable <data_t> ::bypass_attempt_exception();
	return right.name != left.name;
}

template <typename data_t>
bool operator>(const variable <data_t> &right, const variable <data_t> &left)
{
	//if (right.param || left.param)
	//	throw variable <data_t> ::bypass_attempt_exception();
	return right.name > left.name;
}

template <typename data_t>
bool operator<(const variable <data_t> &right, const variable <data_t> &left)
{
	//if (right.param || left.param)
	//	throw variable <data_t> ::bypass_attempt_exception();
	return right.name < left.name;
}

template <typename data_t>
bool operator>=(const variable <data_t> &right, const variable <data_t> &left)
{
	if (right.param || left.param)
		throw variable <data_t> ::bypass_attempt_exception();
	return right.name >= left.name;
}

template <typename data_t>
bool operator<=(const variable <data_t> &right, const variable <data_t> &left)
{
	if (right.param || left.param)
		throw variable <data_t> ::bypass_attempt_exception;
	return right.name <= left.name;
}

/* Derived member functions */
template <typename data_t>
token::type variable <data_t> ::caller() const
{
	return VARIABLE;
}

template <typename data_t>
std::string variable <data_t> ::str() const
{
	std::ostringstream scin;

	scin << "[" << name << "] - ";
	if (param)
		scin << " NULL (PARAMETER)";
	else
		scin << val;

	return scin.str();
} 

#endif
