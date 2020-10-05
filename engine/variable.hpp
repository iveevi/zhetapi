#ifndef VARIALBE_H_
#define VARIALBE_H_

// C/C++ headers
#include <string>
#include <iostream>
#include <sstream>

// Engine headers
#include <token.hpp>
#include <types.hpp>

namespace zhetapi {

	template <class T, class U>
	class Variable : public token {
		std::string	__symbol;
		token		*__tptr;
	public:
		// Constructor
		Variable(const std::string & = "", token * = nullptr);

		template <class A>
		Variable(const std::string &, const A &);

		Variable(const Variable &);

		// Destructor
		~Variable();

		// Copy
		Variable &operator=(const Variable &);

		// Reference
		const token *get() const;
		token *ref();

		const std::string &symbol() const;
		
		// Virtual functions
		token::type caller() const override;
		std::string str() const override;
		token *copy() const;
		bool operator==(token *) const;

		// Comparing functions
		template <class A, class B>
		friend bool operator<(const Variable <A, B> &, const Variable <A, B> &);

		template <class A, class B>
		friend bool operator>(const Variable <A, B> &, const Variable <A, B> &);

		// Printing functions
		template <class A, class B>
		friend std::ostream &operator<<(std::ostream &, const Variable <A, B> &);

		// Exceptions
		class illegal_type {};
	};

	// Constructors
	template <class T, class U>
	Variable <T, U> ::Variable(const std::string &str, token *tptr) : __symbol(str), __tptr(tptr) {}

	template <class T, class U>
	template <class A>
	Variable <T, U> ::Variable(const std::string &str, const A &x) : __symbol(str)
	{
		__tptr = types <T, U> ::convert(x);

		if (!__tptr)
			throw illegal_type();
	}

	template <class T, class U>
	Variable <T, U> ::Variable(const Variable <T, U> &other)
	{
		__tptr = other.__tptr->copy();
		__symbol = other.__symbol;
	}

	// Destructors
	template <class T, class U>
	Variable <T, U> ::~Variable()
	{
		delete __tptr;
	}

	// Copy
	template <class T, class U>
	Variable <T, U> &Variable <T, U>::operator=(const Variable <T, U> &other)
	{
		if (this != &other) {
			__tptr = other.__tptr->copy();
			__symbol = other.__symbol;
		}

		return *this;
	}

	// Reference
	template <class T, class U>
	const token *Variable <T, U> ::get() const
	{
		return __tptr;
	}

	template <class T, class U>
	token *Variable <T, U> ::ref()
	{
		return __tptr;
	}

	template <class T, class U>
	const std::string &Variable <T, U> ::symbol() const
	{
		return __symbol;
	}

	// Virtual functions
	template <class T, class U>
	token::type Variable <T, U> ::caller() const
	{
		return var;
	}

	template <class T, class U>
	std::string Variable <T, U> ::str() const
	{
		return __symbol + " [" + __tptr->str() + "]";
	}

	template <class T, class U>
	token *Variable <T, U> ::copy() const
	{
		return new Variable(__symbol, __tptr);
	}

	template <class T, class U>
	bool Variable <T, U> ::operator==(token *tptr) const
	{
		Variable *var = dynamic_cast <Variable *> (tptr);

		if (!var)
			return true;
		
		return (__symbol == var->__symbol) && (__tptr == var->__tptr);
	}

	// Comparison functions
	template <class T, class U>
	bool operator<(const Variable <T, U> &a, const Variable <T, U> &b)
	{
		return a.symbol() < b.symbol();
	}

	template <class T, class U>
	bool operator>(const Variable <T, U> &a, const Variable <T, U> &b)
	{
		return a.symbol() > b.symbol();
	}

	// Printing
	template <class T, class U>
	std::ostream &operator<<(std::ostream &os, const Variable <T, U> &var)
	{
		os << var.str();

		return os;
	}
	
}

#endif
