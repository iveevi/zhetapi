#ifndef TYPES_H
#define TYPES_H

#include <bits/c++config.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

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

		/* Virtual:
		 * [type] [caller]() - inspector function passed
		 * on to all derived classes */
		virtual type caller();

                /* Virtual:
		 * string [str]() - returns the string
		 * representation of the token */
		virtual std::string str() const = 0;
	};

	token::type token::caller()
	{
		return NONE;
	}
	
	/* Exception Class:
	 *
	 * Represents the generic error that
	 * is thrown whenever an error occurs
	 * when trying to compute with the
	 * operands */
	class exception {
	protected:
		/* std::string [msg] - what the
		 * error is about */
		std::string msg;
	public:
		/* Constructors:
		 * exception() - default constructor
		 *   that sets the private member [msg]
		 *   to an empty string
		 * exception(std::string) - sets the
		 *   private member variable [msg] to
		 *   whatever string is passed */
		exception();
		explicit exception(std::string);

		/* Virtualized Member Functions:
		 * void [set](std::string) - sets the
		 *   private member variable [msg] to
		 *   whatever string is passed
		 * std::string [what]() - returns a constant
		 *   (unchangeable) reference to the contents
		 *   of the private member variable [msg] */
		virtual void set(std::string);
		
		virtual const std::string &what();
	};

	/* Exception Class Member Functions
	 *
	 * See class declaration for a
	 * description of each function
	 *
	 * Constructors: */
	exception::exception() : msg("") {}
	exception::exception(std::string str) : msg(std::move(str)) {}

	/* Virtualized member functions:
	 * setter and getters */
	void exception::set(std::string str)
	{
		msg = std::move(str);
	}

	const std::string &exception::what()
	{
		return msg;
	}

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

                // Add descriptors later
		operand &operator=(const operand &);

		type caller() override;
                std::string str() const override; 

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
	token::type operand <data_t> ::caller()
	{
		return OPERAND;
	}

        template <typename data_t>
        std::string operand <data_t> ::str() const
        {
                return std::to_string(val);
        }

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
		enum order {NA_L0, SA_L1, MDM_L2, FUNC_LMAX};

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

		operation &operator=(const operation &);
		
		type caller() override;
                std::string str() const override;
		
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
		std::cout << "Unimplemented" << std::endl;
		return false;
	}


	template <class oper_t>
	bool operation <oper_t> ::operator[](const std::string &str) const
	{
		std::cout << "Unimplemented" << std::endl;
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
		if (right.pemdas != left.pemdas)
			return false;
		if (right.opers != left.opers)
			return false;
		if (right.name != left.name)
			return false;
		if (right.symbols != left.symbols)
			return false;
		return right.func == left.func;
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
	token::type operation <oper_t> ::caller()
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
		class bypass_attempt_exception : public exception {};

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
		variable(std::string str, bool bl, data_t vl = data_t());

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

                type caller() override;
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
		        variable <data_t> &);

		template <typename type>
		friend std::istream &operator>>(std::istream &is, variable
		        <data_t> &);

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
			throw bypass_attempt_exception("needs change");
		val = vl;
	}

	template <typename data_t>
	data_t &variable <data_t> ::get()
	{
		if (param)
			throw bypass_attempt_exception("needs change");
		return val;
	}

	template <typename data_t>
	const data_t &variable <data_t> ::get() const
	{
		if (param)
			throw bypass_attempt_exception("needs change");
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
			throw bypass_attempt_exception("needs change");
		return par(val, name);
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
			throw bypass_attempt_exception("needs change");
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
			throw bypass_attempt_exception("needs change");
		return pair(val, name);
	}

	template <typename data_t>
	const std::pair <data_t, std::string> &variable <data_t> ::operator*
	                (std::string dstr) const
	{
		if (param)
			throw bypass_attempt_exception("needs change");
		return val;
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
		return right.val == left.val;
	}

	template <typename data_t>
	bool operator!=(const variable <data_t> &right, const variable <data_t> &left)
	{
		if (right.param || left.param)
			throw variable <data_t> ::bypass_attempt_exception;
		return right.val != left.val;
	}

	template <typename data_t>
	bool operator>(const variable <data_t> &right, const variable <data_t> &left)
	{
		if (right.param || left.param)
			throw variable <data_t> ::bypass_attempt_exception;
		return right.val > left.val;
	}

	template <typename data_t>
	bool operator<(const variable <data_t> &right, const variable <data_t> &left)
	{
		if (right.param || left.param)
			throw variable <data_t> ::bypass_attempt_exception;
		return right.val < left.val;
	}

	template <typename data_t>
	bool operator>=(const variable <data_t> &right, const variable <data_t> &left)
	{
		if (right.param || left.param)
			throw variable <data_t> ::bypass_attempt_exception;
		return right.val >= left.val;
	}
	
	template <typename data_t>
	bool operator<=(const variable <data_t> &right, const variable <data_t> &left)
	{
		if (right.param || left.param)
			throw variable <data_t> ::bypass_attempt_exception;
		return right.val <= left.val;
	}

        /* Derived member functions */
	template <typename data_t>
        token::type variable <data_t> ::caller()
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

	/* Beginning of the function class - first complete
	 * toke_tree and ftoken_tree classes
	template <class oper_t>
	class function : public token {
	public:
		typedef oper_t (*ftype)(const std::vector <varible> &);

		function();
		function(std::string, ftype, opers);

		/Use these functions to
		* save space in the function
		* stack class - instead
		* of creating new objects
		* and using space, modify
		* old ones (if the program
		* is sure they wont be used
		* again) *
		void set(std::string);
		void set(function, opers);
		void set(std::string, ftype, opers);

		function get() const;
		function operator~() const;
	private:
		function func;
		std::string name;
		std::size_t opers;
	};

	typedef function <num_t> func_t; */	

	// Beginning of the module class
	template <class oper_t>
	class module : public token {
	private:
		/* The following are the initializations
		 * of the lambda member functions */
		static operation <oper_t> add_op;
		static operation <oper_t> sub_op;
		static operation <oper_t> mult_op;
		static operation <oper_t> div_op;

		/* The following are the functions
		 * correspodning to each of the operations */
		static oper_t add_f(const std::vector <oper_t> &);
		static oper_t sub_f(const std::vector <oper_t> &);
		static oper_t mult_f(const std::vector <oper_t> &);
		static oper_t div_f(const std::vector <oper_t> &);
	public:
		/* The following are static member functions that
		 * give purpose to the tokens
		 *
		 * const token &get_next( std::string, std::size_t):
		 *   returns the next valid token in the passed
		 *   string from the specified index, or throws an
		 *   error if no token was detected
		 */
		static token *get_next(std::string, std::size_t) noexcept(false);
		
		//static std::vector <token *> *get_tokens(std::string);

                // Always check to make sure
                // oper_t is an operand
                type caller() override;
                std::string str() const override;
		
		/* The following is the array containing
		 * all the default operations, and constants
		 * that represents certain things */
		static const int NOPERS = 0x4;
		static const int ADDOP = 0x0;
                static const int SUBOP = 0x1;
                static const int MULTOP = 0x2;
                static const int DIVOP = 0x3;
		
		static operation <oper_t> opers[];
	};

	/* Corresponding functios */
	template <typename oper_t>
	oper_t module <oper_t> ::add_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() + inputs[1].get());
		return new_oper_t;
	}

        template <typename oper_t>
	oper_t module <oper_t> ::sub_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() - inputs[1].get());
		return new_oper_t;
	}

        template <typename oper_t>
	oper_t module <oper_t> ::mult_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() * inputs[1].get());
		return new_oper_t;
	}

        template <typename oper_t>
	oper_t module <oper_t> ::div_f(const std::vector <oper_t> &inputs)
	{
		oper_t new_oper_t = oper_t(inputs[0].get() / inputs[1].get());
		return new_oper_t;
	}

	// Module's default operations
	template <typename oper_t>
	operation <oper_t> module <oper_t> ::add_op = operation <oper_t>
	(std::string {"add_op"}, module <oper_t> ::add_f, 2, std::vector
        <std::string> {"+", "plus", "add"}, operation <oper_t>::SA_L1,
	std::vector <std::string> {"8+8"});

        template <typename oper_t>
	operation <oper_t> module <oper_t> ::sub_op = operation <oper_t>
	(std::string {"sub_op"}, module <oper_t> ::sub_f, 2, std::vector
        <std::string> {"-", "minus"}, operation <oper_t>::SA_L1,
	std::vector <std::string> {"8-8"});

        template <typename oper_t>
	operation <oper_t> module <oper_t> ::mult_op = operation <oper_t>
	(std::string {"mult_op"}, module <oper_t> ::mult_f, 2, std::vector
        <std::string> {"*", "times", "by"}, operation <oper_t>::MDM_L2,
	std::vector <std::string> {"8*8"});

        template <typename oper_t>
	operation <oper_t> module <oper_t> ::div_op = operation <oper_t>
	(std::string {"div_op"}, module <oper_t> ::add_f, 2, std::vector
        <std::string> {"/", "divided by"}, operation <oper_t>::MDM_L2,
	std::vector <std::string> {"8/8"});

	template <typename oper_t>
	operation <oper_t> module <oper_t> ::opers[] = {
		add_op, sub_op, mult_op, div_op,
	};
	
	typedef module <num_t> module_t;

        // Module's parsing functions
        template <typename oper_t>
        token *get_next()

        // Derived member functions
        template <typename oper_t>
        token::type module <oper_t> ::caller()
        {
                return MODULE;
        }

        template <typename oper_t>
        std::string module <oper_t> ::str() const
        {
                // Add some more description
                // to the returned string
                return "module";
        }
}

#endif
