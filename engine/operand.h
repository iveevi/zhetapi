#ifndef OPERAND_H_
#define OPERAND_H_

// Custom Built Libraries
#include "token.h"

/* Operand Class:
 * 
 * Represents an operand in mathematics
 * by using data_t as the numerical data
 * type or value */
template <typename data_t>
class operand : public token {
	/* data_t [val] - the only member of operand
	 * which represents its value */
	data_t val;
public:
	/* Constructors:
	 * operand() - sets the private member variable
	 *   [val] to the default value of data_t
	 * operand(data_t) - sets the private member variable
	 *   [val] to whatever value is passed */
	operand();
	explicit operand(data_t);
	operand(const operand &);

	/* Virtualized Member Functions:
	 * void [set](data_t) - sets the private member variable
	 *   to whatever value is passed
	 * void operator[](data_t) - sets the private member
	 *   variable to whatever value is passed
	 * data_t &[get]() - returns a reference to the private
	 *   member variable
	 * const data_t &[get]() - returns a constant (unchangeable)
	 *   reference to the private member variable
	 * data_t &operator*() - returns a reference to the private
	 *   member variable
	 * const data_t &operator*() - returns a constant (unchangeable)
	 *   reference to the private member variable */
	virtual void set(data_t);
	virtual void operator[](data_t);
	
	virtual data_t &get();
	virtual const data_t &get() const;

	virtual data_t &operator*();
	virtual const data_t &operator*() const;

	const std::string &symbol() const {
		std::string *nout = new std::string(str());
		return *nout;
	}

	// Add descriptors later
	operand &operator=(const operand &);

	type caller() const override;
	std::string str() const override;
	token *copy() const override;

	bool operator==(token *) const override;

	/* Friends:
	 * std::ostream &operator<<(std::ostream &, const operand
	 *   <data_t> &) - outputs the value of val onto the stream
	 *   pointed to by the passed ostream object
	 * std::istream &operator>>(std::istream &, operand &) - reads
	 *   input from the stream passed in and sets the value
	 *   of the val in the passed operand object to the read data_t
	 *   value */
	template <typename type>
	friend std::ostream &operator<<(std::ostream &, const
		operand <data_t> &);

	template <typename type>
	friend std::istream &operator>>(std::istream &, operand
		<data_t> &);

	/* Comparison Operators: */
	template <typename type>
	friend bool &operator==(const operand &, const operand &);
	
	template <typename type>
	friend bool &operator!=(const operand &, const operand &);
	
	template <typename type>
	friend bool &operator>(const operand &, const operand &);
	
	template <typename type>
	friend bool &operator<(const operand &, const operand &);
	
	template <typename type>
	friend bool &operator>=(const operand &, const operand &);
	
	template <typename type>
	friend bool &operator<=(const operand &, const operand &);
};

/* Default operand specification
 * using the default data/info type */
typedef operand <def_t> num_t;

/* Operand Class Member Functions
 * 
 * See class declaration to see a
 * description of each function
 *
 * Constructors: */
template <typename data_t>
operand <data_t> ::operand () : val(data_t()) {}

template <typename data_t>
operand <data_t> ::operand(data_t data) : val(data) {}

template <typename data_t>
operand <data_t> ::operand(const operand &other) : val(other.val) {}

/* Virtualized member functions:
 * setters, getter and operators */
template <typename data_t>
void operand <data_t> ::set(data_t data)
{
	val = data;
}

template <typename data_t>
void operand <data_t> ::operator[](data_t data)
{
	val = data;
}

template <typename data_t>
data_t &operand <data_t> ::get()
{
	return val;
}

template <typename data_t>
const data_t &operand <data_t> ::get() const
{
	return val;
}

template <typename data_t>
data_t &operand <data_t> ::operator*()
{
	return val;
}

template <typename data_t>
const data_t &operand <data_t> ::operator*() const
{
	return val;
}

template <typename data_t>
operand <data_t> &operand <data_t> ::operator=(const operand &other)
{
	val = other.val;
	return *this;
}

/* Friend functions: istream and
 * ostream utilities */
template <typename data_t>
std::ostream &operator<< (std::ostream &os, const operand <data_t>
	&right)
{
	os << right.get();
	return os;
}

template <typename data_t>
std::istream &operator>> (std::istream &is, operand <data_t> &right)
{
	data_t temp;
	is >> temp;
	right.set(temp);
	return is;
}

/* Comparison functions: */
template <typename data_t>
bool operator==(const operand <data_t> &right, const operand
	<data_t> &left)
{
	return *right == *left;
}

template <typename data_t>
bool operator!=(const operand <data_t> &right, const operand
	<data_t> &left)
{
	return right != left;
}

template <typename data_t>
bool operator>(const operand <data_t> &right, const operand <data_t>
	&left)
{
	return right.val > left.val;
}

template <typename data_t>
bool operator<(const operand <data_t> &right, const operand <data_t>
&left)
{
	return right.val < left.val;
}

template <typename data_t>
bool operator>=(const operand <data_t> &right, const operand <data_t>
&left)
{
	return right.val >= left.val;

}
template <typename data_t>
bool operator<=(const operand <data_t> &right, const operand <data_t>
&left)
{
	return right.val <= left.val;
}

/* Token class derived functions: */
template <typename data_t>
token::type operand <data_t> ::caller() const
{
	return OPERAND;
}

template <typename data_t>
std::string operand <data_t> ::str() const
{
	return std::to_string(val);
}

template <class T>
token *operand <T> ::copy() const
{
	return new operand(*this);
}

template <class T>
bool operand <T> ::operator==(token *t) const
{
	if (t->caller() != token::OPERAND)
		return false;

	return val == (dynamic_cast <operand *> (t))->get();
}

#endif
