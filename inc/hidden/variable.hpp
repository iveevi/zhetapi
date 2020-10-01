#ifndef VARIALBE_H_
#define VARIALBE_H_

// C/C++ headers
#include <string>
#include <iostream>
#include <sstream>

// Engine headers
#include <token.hpp>

namespace zhetapi {

	/**
	 * @brief Represents a variable in mathematics which can be used as a
	 * dummy variable for custom functions and has the safe ability to read
	 * and write the value it stores.
	 *
	 * @tparam data_t The type of data the variable holds.
	 */
	template <typename data_t>
	class Variable : public token {
		/* std::string [name] - the name of the
		 * Variable that will be taken into account
		 * when creating custom functions and using
		 * them later on */
		std::string name;

		/* data_t [val] - the value stored by the
		 * object which serves the same purpose as
		 * the val member in the operand class */
		data_t val;

		/* bool [param] - a boolean Variable that
		 * represents whether or not the current
		 * Variable object is being used a parameter
		 * (dummy) or to hold information */
		bool param;
	public:
		/*
		 * Bypass_Attempt_Exception Class:
		 *
		 * Represents the error thrown whenever
		 * the value of the Variable object is
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
		 * Variable() - creates a new Variable object
		 *   that is by default a dummy parameter
		 * Variable(std::string, boolean, data_t) - creates
		 *   a new Variable object with corresponding
		 *   members with respect to the passed parameters,
		 *   private member Variable [val] is defaulted
		 *   to the default value of data_t */
		Variable();
		Variable(const std::string &, const data_t & = data_t());

		// change to const refs
		Variable(std::string, bool, data_t = data_t());

		Variable(const Variable &);

		/* Virtualized Member Functions:
		 * void [set](bool) - changes whether or not the
		 *   current Variable object acts as a dummy
		 *   or as a value
		 * void [set](data_t) - sets the private member
		 *   Variable [val] to the value that is passed
		 * void [set](std::string) - sets the private
		 *   member Variable [name] to the string
		 *   that is passed
		 * void [set](data_t, std::string) - sets
		 *   the name, [name], and value, [val], of
		 *   the current Variable object
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
		 *   of the current Variable object
		 * const std::pair <data_t, std::string> &[get]()
		 *   const - returns a tuple of the value and name of
		 *   the current Variable object, or throws a bypass
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

		bool is_param() const {return param;}

		// returns the name of the Variable
		virtual const std::string &symbol() const;

		virtual data_t ttype() const {
			return data_t();
		}

		type caller() const override;
		std::string str() const override;
		token *copy() const override;

		bool operator==(token *) const override;
		
		/* Friends:
		 * std::ostream &operator<<(std::ostream &, const Variable
		 *   <data_t> &) - outputs the name of the passed Variable
		 *   object, its value, and whether or not it is a dummy
		 * std::istream &operator>>(std::istream &, Variable &) - reads
		 *   information from the passed istream object, and modifies
		 *   the passed Variable object appropriately */
		template <typename type>
		friend std::ostream &operator<<(std::ostream &os, const
			Variable <type> &);

		template <typename type>
		friend std::istream &operator>>(std::istream &is, Variable
			<type> &);

		/* Comparison Functions:
		 *
		 * The following comparison operators
		 * execute by comparing the values of
		 * the Variables, and throw a bypass
		 * attempt error if any of the Variables
		 * are parameters */
		template <typename type>
		friend bool operator==(const Variable <type> &, const
			Variable <type> &) noexcept(false);

		template <typename type>
		friend bool operator!=(const Variable <type> &, const
			Variable <type> &) noexcept(false);

		template <typename type>
		friend bool operator>(const Variable <type> &, const
			Variable <type> &) noexcept(false);

		template <typename type>
		friend bool operator<(const Variable <type> &, const
			Variable <type> &) noexcept(false);

		template <typename type>
		friend bool operator>=(const Variable <type> &, const
			Variable <type> &) noexcept(false);

		template <typename type>
		friend bool operator<=(const Variable <type> &, const
			Variable <type> &) noexcept(false);

		// on
		template <class A>
		operator Variable <A> ();
	};

	/* Variable Class Member Functions
	 *
	 * See class declaration for a
	 * description of each function
	 *
	 * Constructors: */
	template <typename data_t>
	Variable <data_t> ::Variable() : val(data_t()), name("x"),
		param(false) {}

	template <class T>
	Variable <T> ::Variable(const std::string &str, const T &dt)
		: val(dt), name(str), param(false) {}

	template <typename data_t>
	Variable <data_t> ::Variable(std::string str, bool bl, data_t vl)
		: val(vl), name(str), param(bl) {}

	template <class T>
	Variable <T> ::Variable(const Variable &other) : name(other.name),
		val(other.val), param(other.param) {}

	/* Virtualized member functions:
	 * setters, getters and operators */
	template <typename data_t>
	void Variable <data_t> ::set(bool bl)
	{
		param = bl;
	}

	template <typename data_t>
	void Variable <data_t> ::set(data_t vl)
	{
		val = vl;
	}

	template <typename data_t>
	void Variable <data_t> ::set(std::string str)
	{
		name = str;
	}

	template <typename data_t>
	void Variable <data_t> ::set(data_t vl, std::string str)
	{
		val = vl;
		name = str;
	}

	template <typename data_t>
	void Variable <data_t> ::operator[](bool bl)
	{
		param = bl;
	}

	template <typename data_t>
	void Variable <data_t> ::operator[](data_t vl)
	{
		if (param)
			throw bypass_attempt_exception();
		val = vl;
	}

	template <typename data_t>
	data_t &Variable <data_t> ::get()
	{
		if (param)
			throw bypass_attempt_exception();
		return val;
	}

	template <typename data_t>
	const data_t &Variable <data_t> ::get() const
	{
		if (param)
			throw bypass_attempt_exception();
		return val;
	}

	template <typename data_t>
	const std::string &Variable <data_t> ::get(int dm) const
	{
		return name;
	}

	template <typename data_t>
	const std::pair <data_t, std::string> &Variable <data_t> ::get
			(std::string dstr) const
	{
		if (param)
			throw bypass_attempt_exception();
		return *(new std::pair <data_t, std::string> (val, name));
	}

	template <typename data_t>
	void Variable <data_t> ::operator[](std::string str)
	{
		name = str;
	}

	template <typename data_t>
	const data_t &Variable <data_t> ::operator*() const
	{
		if (param)
			throw bypass_attempt_exception();
		return val;
	}

	template <typename data_t>
	const std::string &Variable <data_t> ::operator*(int dm) const
	{
		return name;
	}

	template <typename data_t>
	data_t &Variable <data_t> ::operator*()
	{
		if (param)
			throw bypass_attempt_exception();
		return val;
	}

	template <typename data_t>
	const std::pair <data_t, std::string> &Variable <data_t> ::operator*
			(std::string dstr) const
	{
		if (param)
			throw bypass_attempt_exception();
		return *(new std::pair <data_t, std::string> (val, name));
	}

	template <typename data_t>
	const std::string &Variable <data_t> ::symbol() const
	{
		return name;
	}

	/* Friend functions: ostream and
	 * istream utilities */
	template <typename data_t>
	std::ostream &operator<<(std::ostream &os, const Variable <data_t> &var)
	{
		os << "[" << var.name << "] - ";

		if (var.param)
			os << " NULL (PARAMETER)";
		else
			os << var.val;

		return os;
	}

	template <typename data_t>
	std::istream &operator>>(std::istream &is, Variable <data_t> &var)
	{
		// Implement in the general scope
		// Later, after trees
		return is;
	}

	/* Comparison functions: execute comparison
	 * functions by comparing values, and throws
	 * a bypass error if one argument is a parameter */
	template <typename data_t>
	bool operator==(const Variable <data_t> &right, const Variable <data_t> &left)
	{
		// Later, add constructor for
		// Bypass exception that takes in
		// The name of the violated Variable
		
		if (right.param || left.param) // Distinguish later
			throw typename Variable <data_t> ::bypass_attempt_exception();
		return right.name == left.name;
	}

	template <typename data_t>
	bool operator!=(const Variable <data_t> &right, const Variable <data_t> &left)
	{
		if (right.param || left.param)
			throw Variable <data_t> ::bypass_attempt_exception();
		return right.name != left.name;
	}

	template <typename data_t>
	bool operator>(const Variable <data_t> &right, const Variable <data_t> &left)
	{
		//if (right.param || left.param)
		//	throw Variable <data_t> ::bypass_attempt_exception();
		return right.name > left.name;
	}

	template <typename data_t>
	bool operator<(const Variable <data_t> &right, const Variable <data_t> &left)
	{
		//if (right.param || left.param)
		//	throw Variable <data_t> ::bypass_attempt_exception();
		return right.name < left.name;
	}

	template <typename data_t>
	bool operator>=(const Variable <data_t> &right, const Variable <data_t> &left)
	{
		if (right.param || left.param)
			throw Variable <data_t> ::bypass_attempt_exception();
		return right.name >= left.name;
	}

	template <typename data_t>
	bool operator<=(const Variable <data_t> &right, const Variable <data_t> &left)
	{
		if (right.param || left.param)
			throw Variable <data_t> ::bypass_attempt_exception;
		return right.name <= left.name;
	}

	/* Derived member functions */
	template <typename data_t>
	token::type Variable <data_t> ::caller() const
	{
		return var;
	}

	template <typename data_t>
	std::string Variable <data_t> ::str() const
	{
		std::ostringstream scin;

		scin << "[" << name << "] - ";
		if (param)
			scin << " NULL (PARAMETER)";
		else
			scin << val;

		return scin.str();
	}

	template <class T>
	token *Variable <T> ::copy() const
	{
		return new Variable(name, param, val);
	}

	template <class T>
	bool Variable <T> ::operator==(token *t) const
	{
		if (t->caller() != token::var)
			return false;

		return name == (dynamic_cast <Variable *> (t))->symbol();
	}

	/*
	 * Conversion between variables as a casting of their values. Required
	 * for compilation purposes in the Barn class.
	 */
	template <class T>
	template <class A>
	Variable <T> ::operator Variable <A> ()
	{
		/*
		 * No need to actually retrieve the value of the value and cast
		 * it. This is simlply a dummy method which will make
		 * conversions legal.
		 */

		return Variable <A> {name, param, A()};
	}

}

#endif
