#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace types {
	// Default numerical value
	typedef double def_t;

	// Dummy class for convenience;
	class token {};

	// Operand class declaration
	template <typename data_t>
	class operand : public token {
		data_t val;
	public:
		operand();
		operand(data_t);

		void set(data_t);

		void operator[] (data_t);

		data_t &get();
		const data_t &get() const;

		data_t &operator~ ();
		const data_t &operator~ () const;

		template <typename type>
		friend std::ostream &operator<< (std::ostream &os, const operand <data_t> &);

		template <typename type>
		friend std::istream &operator>> (std::istream &is, operand <data_t> &);
	};

	typedef operand <def_t> num_t;
	
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

	// Beginning of the variable class
	template <typename data_t>
	class variable : public token {
		std::string name;
		data_t val;
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

	// Default Operations
	template <typename oper_t>
	operation <oper_t> add_op = operation <oper_t>
	("add_op",[](const std::vector <oper_t> &inputs) {
		return oper_t(inputs[0].get() + inputs[1].get());
	}, 2, {"+", "plus", "add"});

	template <typename oper_t>
	operation <oper_t> sub_op = operation <oper_t>
	("sub_op", [](const std::vector <oper_t> &inputs) {
		return oper_t(inputs[0].get() - inputs[1].get());
	}, 2, {"-", "minus", "subtract"});

	template <typename oper_t>
	operation <oper_t> mult_op = operation <oper_t>
	("mult_op", [](const std::vector <oper_t> &inputs) {
		return oper_t(inputs[0].get() * inputs[1].get());
	}, 2, {"*", "mult", "times"});

	template <typename oper_t>
	operation <oper_t> div_op = operation <oper_t>
	("div_op", [](const std::vector <oper_t> &inputs) {
		return oper_t(inputs[0].get() / inputs[1].get());
	}, 2, {"/", "divided by"});
}

#include "types.cpp"

#endif
