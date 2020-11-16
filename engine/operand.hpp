#ifndef Operand_H_
#define Operand_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include "token.hpp"

namespace zhetapi {

	/* Operand Class:
	 * 
	 * Represents an Operand in mathematics
	 * by using data_t as the numerical data
	 * type or value */
	template <typename data_t>
	class Operand : public Token {
		/* data_t [val] - the only member of Operand
		 * which represents its value */
		data_t val;
	public:
		/* Constructors:
		 * Operand() - sets the private member Variable
		 *   [val] to the default value of data_t
		 * Operand(data_t) - sets the private member Variable
		 *   [val] to whatever value is passed */
		Operand();
		// Operand(data_t);
		Operand(const data_t &);
		Operand(const Operand &);

		/* Virtualized Member Functions:
		 * void [set](data_t) - sets the private member Variable
		 *   to whatever value is passed
		 * void operator[](data_t) - sets the private member
		 *   Variable to whatever value is passed
		 * data_t &[get]() - returns a reference to the private
		 *   member Variable
		 * const data_t &[get]() - returns a constant (unchangeable)
		 *   reference to the private member Variable
		 * data_t &operator*() - returns a reference to the private
		 *   member Variable
		 * const data_t &operator*() - returns a constant (unchangeable)
		 *   reference to the private member Variable */
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
		Operand &operator=(const Operand &);

		type caller() const override;
		std::string str() const override;
		Token *copy() const override;

		bool operator==(Token *) const override;

		/* Friends:
		 * std::ostream &operator<<(std::ostream &, const Operand
		 *   <data_t> &) - outputs the value of val onto the stream
		 *   pointed to by the passed ostream object
		 * std::istream &operator>>(std::istream &, Operand &) - reads
		 *   input from the stream passed in and sets the value
		 *   of the val in the passed Operand object to the read data_t
		 *   value */
		template <typename type>
		friend std::ostream &operator<<(std::ostream &, const
			Operand <data_t> &);

		template <typename type>
		friend std::istream &operator>>(std::istream &, Operand
			<data_t> &);

		/* Comparison Operators: */
		template <typename type>
		friend bool &operator==(const Operand &, const Operand &);
		
		template <typename type>
		friend bool &operator!=(const Operand &, const Operand &);
		
		template <typename type>
		friend bool &operator>(const Operand &, const Operand &);
		
		template <typename type>
		friend bool &operator<(const Operand &, const Operand &);
		
		template <typename type>
		friend bool &operator>=(const Operand &, const Operand &);
		
		template <typename type>
		friend bool &operator<=(const Operand &, const Operand &);

		// on
		template <class A>
		operator Operand <A> ();
	};

	/* Operand Class Member Functions
	 * 
	 * See class declaration to see a
	 * description of each function
	 *
	 * Constructors: */
	template <typename data_t>
	Operand <data_t> ::Operand () : val(data_t()) {}

	//template <typename data_t>
	//Operand <data_t> ::Operand(data_t data) : val(data) {}
	
	template <typename data_t>
	Operand <data_t> ::Operand(const data_t &data) : val(data) {}

	template <typename data_t>
	Operand <data_t> ::Operand(const Operand &other) : val(other.val) {}

	/* Virtualized member functions:
	 * setters, getter and operators */
	template <typename data_t>
	void Operand <data_t> ::set(data_t data)
	{
		val = data;
	}

	template <typename data_t>
	void Operand <data_t> ::operator[](data_t data)
	{
		val = data;
	}

	template <typename data_t>
	data_t &Operand <data_t> ::get()
	{
		return val;
	}

	template <typename data_t>
	const data_t &Operand <data_t> ::get() const
	{
		return val;
	}

	template <typename data_t>
	data_t &Operand <data_t> ::operator*()
	{
		return val;
	}

	template <typename data_t>
	const data_t &Operand <data_t> ::operator*() const
	{
		return val;
	}

	template <typename data_t>
	Operand <data_t> &Operand <data_t> ::operator=(const Operand &other)
	{
		val = other.val;
		return *this;
	}

	/* Friend functions: istream and
	 * ostream utilities */
	template <typename data_t>
	std::ostream &operator<< (std::ostream &os, const Operand <data_t>
		&right)
	{
		os << right.get();
		return os;
	}

	template <typename data_t>
	std::istream &operator>> (std::istream &is, Operand <data_t> &right)
	{
		data_t temp;
		is >> temp;
		right.set(temp);
		return is;
	}

	/* Comparison functions: */
	template <typename data_t>
	bool operator==(const Operand <data_t> &right, const Operand
		<data_t> &left)
	{
		return *right == *left;
	}

	template <typename data_t>
	bool operator!=(const Operand <data_t> &right, const Operand
		<data_t> &left)
	{
		return right != left;
	}

	template <typename data_t>
	bool operator>(const Operand <data_t> &right, const Operand <data_t>
		&left)
	{
		return right.val > left.val;
	}

	template <typename data_t>
	bool operator<(const Operand <data_t> &right, const Operand <data_t>
	&left)
	{
		return right.val < left.val;
	}

	template <typename data_t>
	bool operator>=(const Operand <data_t> &right, const Operand <data_t>
	&left)
	{
		return right.val >= left.val;

	}
	template <typename data_t>
	bool operator<=(const Operand <data_t> &right, const Operand <data_t>
	&left)
	{
		return right.val <= left.val;
	}

	/* Token class derived functions: */
	template <typename data_t>
	Token::type Operand <data_t> ::caller() const
	{
		return opd;
	}

	template <typename data_t>
	std::string Operand <data_t> ::str() const
	{
		std::ostringstream oss;

		oss << val;

		return oss.str();
	}

	template <class T>
	Token *Operand <T> ::copy() const
	{
		return new Operand(*this);
	}

	template <class T>
	bool Operand <T> ::operator==(Token *t) const
	{
		if (dynamic_cast <Operand <T> *> (t) == nullptr)
			return false;

		return val == (dynamic_cast <Operand *> (t))->get();
	}

	template <class T>
	template <class A>
	Operand <T> ::operator Operand <A> ()
	{
		return Operand <A> {(A) val};
	}

}

#endif
