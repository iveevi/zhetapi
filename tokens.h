#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <string>
#include <utility>

namespace tokens {
	/* Default data/info type to
	 * use for calculations */
	typedef double def_t;

	/* Token Class:
	 *
	 * Acts as a dummy class for
	 * use of generic pointer in
	 * other modules */
	class token {
	public:
		/* Enumerations:
		 * [type] - new data type to allow function
		 * caller inspection */
		enum type {NONE, OPERAND, OPERATION,
			VARIABLE, FUNCTION, MODULE};

		/* Virtuals:
		 * [type] [caller]() - insepctor function passed
		 * on to all derived classes */
		virtual type caller() = 0;
	};

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
		 *   val to the default value of data_t
		 * operand(data_t) - sets the private member variable
		 *   to whatever value is passed */
		operand();
		operand(data_t);

		/* Regular Member Functions:
		 * void [set](data_t) - sets the private member variable
		 *   to whatever value is passed
		 * void operator[](data_t) - sets the private member
		 *   variable to whatever value is passed
		 * data_t &[get]() - returns a reference to the private
		 *   member variable
		 * const data_t &[get]() - returns a constant (unchangable)
		 *   reference to the prviate member variable
		 * data_t &operator*() - returns a reference to the private
		 *   member variable
		 * const data_t &operator*() - returns a constant (unchangable)
		 *   reference to the private member variable */
		void set(data_t);
		void operator[](data_t);
		
		data_t &get();
		const data_t &get() const;

		data_t &operator*();
		const data_t &operator*() const;

		/* Friends:
		 * std::ostream &operator<< (std::ostream &, const operand
		 *   <data_t> &) - outputs the value of val onto the stream
		 *   pointed to by the passed ostream object
		 * std::istream &operator>> (std::istream &, operand &) - reads
		 *   input from the stream passed in and sets the value
		 *   of the val in the passed operand object to the read data_t
		 *   value */
		template <typename type>
		friend std::ostream &operator<< (std::ostream &os, const operand <data_t> &);

		template <typename type>
		friend std::istream &operator>> (std::istream &is, operand <data_t> &);
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
	operand <data_t> ::operand(data_t nval)
	{
		set(nval);
	}

	/* Regular member functions:
	 * setters, getter and operators */
	template <typename data_t>
	void operand <data_t> ::set(data_t nval)
	{
		val = nval;
	}

	template <typename data_t>
	void operand <data_t> ::operator[](data_t nval)
	{
		val = nval;
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

	/* Friend functions: istream and ostream utilities */
	template <typename data_t>
	std::ostream &operator<< (std::ostream &os, const operand <data_t> &right)
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

	// Beginning of the operation class
	template <class oper_t>
	class operation : public token {
	protected:
		// Exception Classes
		class exception {
		public:
			std::string msg;

			exception();
			exception(std::string);

			virtual void set(std::string);
		};

		class argset_exception : public exception {
		public:
			argset_exception();
			argset_exception(int, const operation <oper_t> &);
			argset_exception(std::string);
			argset_exception(std::string, int, int);

			void set(int, const operation <oper_t> &);
			void set(std::string);
			void set(std::string, int, int);
		};

		class computation_exception : public exception {};
	public:
		// Typedefs
		typedef oper_t (*function)(const std::vector <oper_t> &);

		// Member Functions
		operation();
		operation(std::string, function, int, const std::vector <std::string> &);

		void set(std::string, function, int, const std::vector <std::string> &);

		function get() const;
		function operator~() const;

		oper_t compute(const std::vector <oper_t> &) const noexcept(false);
		oper_t operator() (const std::vector <oper_t> &) const noexcept(false);  
	private:
		// Real Members
		function func;
		std::string name;
		std::size_t opers;
		std::vector <std::string> symbols;
	};

	typedef operation <num_t> opn_t;

	// Operation Member Functions
	template <class oper_t>
	operation <oper_t> ::operation()
	{
		func = nullptr;
	}

	template <class oper_t>
	operation <oper_t> ::operation(std::string str, function nfunc, int nopers,
		const std::vector <std::string> &nsymbols)
	{
		name = str;
		func = nfunc;
		opers = nopers;
		symbols = nsymbols;
	}

	template <class oper_t>
	void operation <oper_t> ::set(std::string str, function nfunc, int nopers,
		const std::vector <std::string> &nsymbols)
	{
		name = str;
		func = nfunc;
		opers = nopers;
		symbols = nsymbols;
	}

	template <class oper_t>
	typename operation <oper_t>::function operation <oper_t> ::get() const
	{
		return func;
	}

	template <class oper_t>
	typename operation <oper_t>::function operation <oper_t> ::operator~ () const
	{
		return func;
	}

	template <class oper_t>
	oper_t operation <oper_t> ::compute(const std::vector <oper_t> &inputs) const
		noexcept(false)
	{
		if (inputs.size() != opers)
			throw argset_exception(inputs.size(), *this);
		return (*func)(inputs);
	}

	template <class oper_t>
	oper_t operation <oper_t> ::operator() (const std::vector <oper_t> &inputs) const
		noexcept(false)
	{
		if (inputs.size() != opers)
			throw argset_exception(inputs.size(), *this);
		return (*func)(inputs);
	}

	// Exception (Base Exception Class) Implementation
	template <class oper_t>
	operation <oper_t> ::exception::exception() : msg("") {}

	template <class oper_t>
	operation <oper_t> ::exception::exception(std::string str) : msg(str) {}

	template <class oper_t>
	void operation <oper_t> ::exception::set(std::string str) {msg = str;}

	// Argset Exception (Derived Class From Exception) Implementation
	template <class oper_t>
	operation <oper_t> ::argset_exception::argset_exception()
		: operation <oper_t> ::exception() {}

	template <class oper_t>
	operation <oper_t> ::argset_exception::argset_exception(int actual, const operation <oper_t> &obj)
	{
		using std::to_string;
		exception::msg = obj.name + ": Expected " + to_string(obj.opers);
		exception::msg += " operands, received " + to_string(actual) + " instead.";
	}

	template <class oper_t>
	operation <oper_t> ::argset_exception::argset_exception(std::string str)
		: operation <oper_t> ::exception(str) {}

	template <class oper_t>
	operation <oper_t> ::argset_exception::argset_exception(std::string str, int expected, int actual)
	{
		using std::to_string;
		exception::msg = str + ": Expected " + to_string(expected);
		exception::msg += " operands, received " + to_string(actual) + "instead.";
	}

	template <class oper_t>
	void operation <oper_t> ::argset_exception::set(int actual, const operation <oper_t> &obj)
	{
		using std::to_string;
		exception::msg = obj.name + ": Expected " + to_string(obj.opers);
		exception::msg += " operands, received " + to_string(actual) + " instead.";
	}

	template <class oper_t>
	void operation <oper_t> ::argset_exception::set(std::string str) {exception::msg = str;}

	template <class oper_t>
	void operation <oper_t> ::argset_exception::set(std::string str, int expected, int actual)
	{
		using std::to_string;
		exception::msg = str + ": Expected " + to_string(expected);
		exception::msg += " operands, received " + to_string(actual) + "instead.";
	}

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
		variable();
		variable(std::string str, bool par, data_t data = data_t());

		void set(data_t);
		void set(std::string);
		void set(data_t, std::string);

		void operator[] (data_t);

		std::pair <data_t, std::string> &get() const;
		std::pair <data_t, std::string> &operator~() const;
	};

	typedef variable <def_t> var_t;

	// Beginning of the function class
	template <class oper_t>
	class function : public operation {
	public:
		typedef oper_t (*function)(const std::vector <varible> &);

		function();
		function(std::string, function, opers);

		/* Use these functions to
		* save space in the function
		* stack class - instead
		* of creating new objects
		* and using space, modify
		* old ones (if the program
		* is sure they wont be used
		* again)
		*/
		void set(std::string);
		void set(function, opers);
		void set(std::string, function, opers);

		function get() const;
		function operator~() const;
	private:
		function func;
		std::string name;
		std::size_t opers;
	};

	typedef function <num_t> func_t;	

	// Beginning of the module class
	template <class oper_t>
	class module : public operation {
	private:
		/* The following are the initializations
		 * of the lambda member functions */
		operation <oper_t> add_op = operation <oper_t>
		("add_op",[](const std::vector <oper_t> &inputs) {
			return oper_t(inputs[0].get() + inputs[1].get());
		}, 2, {"+", "plus", "add"});

		operation <oper_t> sub_op = operation <oper_t>
		("sub_op", [](const std::vector <oper_t> &inputs) {
			return oper_t(inputs[0].get() - inputs[1].get());
		}, 2, {"-", "minus", "subtract"});

		operation <oper_t> mult_op = operation <oper_t>
		("mult_op", [](const std::vector <oper_t> &inputs) {
			return oper_t(inputs[0].get() * inputs[1].get());
		}, 2, {"*", "mult", "times"});
		
		operation <oper_t> div_op = operation <oper_t>
		("div_op", [](const std::vector <oper_t> &inputs) {
			return oper_t(inputs[0].get() / inputs[1].get());
		}, 2, {"/", "divided by"});
	public:
		/* The following are static member functions that
		 * give purpose to the tokens
		 *
		 * const token &get_next( std::string, std::size_t):
		 *   returns the next valid token in the passed
		 *   string from the specified index, or throws an
		 *   error if no token was detected
		 */
		static const token &get_next(std::string, std::size_t) noexcept(false);
		static vector <token *> *get_tokens(std::string);
		
		/* The following is the array containing
		 * all the default operations */
		static const int NOPERS = 0x4;
		static const operation <oper_t> opers {
		   add_op, sub_op, mult_op, div_op,
		};
	};

	typedef module <num_t> module_t;
}

#include "types.cpp"

#endif
