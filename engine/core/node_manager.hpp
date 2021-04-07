#ifndef NODE_MANAGER_H_
#define NODE_MANAGER_H_

// C/C++ headers
#include <fstream>
#include <stack>
#include <set>
#include <vector>

// Engine headers
#include <core/node_reference.hpp>
#include <core/node_differential.hpp>
#include <core/parser.hpp>
#include <core/types.hpp>
#include <core/lvalue.hpp>
#include <core/rvalue.hpp>

namespace zhetapi {

class Engine;
class Function;

class node_manager {
private:
	// TODO: refactor params to args
	// Engine *			_engine	= nullptr;
	node				_tree	= node();
	std::vector <node>		_refs	= {};
	std::vector <std::string>	_params	= {};
public:
	node_manager();
	node_manager(const node_manager &);
	node_manager(Engine *, const node &);
	node_manager(Engine *, const node &,
			const std::vector <std::string> &);
	node_manager(Engine *, const std::string &);

	// TODO: try to merge these two constructors with def. args
	node_manager(Engine *, const std::string &,
			const std::vector <std::string> &,
			const std::set <std::string> & = std::set <std::string> ());

	node_manager &operator=(const node_manager &);

	// Properties
	bool empty() const;	// Is the _tree node empty?
	size_t num_args() const;
	const node &get_tree() const;

	// Setters
	void set_label(lbl);
	// void set_engine(Engine *);

	// Methods
	void unpack();

	Token *value(Engine *) const;
	Token *sequential_value(Engine *) const;
	Token *substitute_and_compute(Engine *, std::vector <Token *> &);
	Token *substitute_and_seq_compute(Engine *, const std::vector <Token *> &);

	void append_front(const node &);
	void append_front(const node_manager &);

	void append(const node &);
	void append(const node_manager &);

	void add_args(const std::vector <std::string> &);

	/*
	 * Responsible for expanding variable clusters and truning them
	 * into product of Operands.
	 *
	 * TODO: make private
	 */
	void expand(Engine *, node &,
			const std::set <std::string> & = std::set <std::string> ());

	void simplify(Engine *);

	void differentiate(const std::string &);

	void refactor_reference(const std::string &, Token *);

	std::string display() const;

	void print(bool = false) const;

	// Arithmetic
	friend node_manager operator+(
			const node_manager &,
			const node_manager &);
	friend node_manager operator-(
			const node_manager &,
			const node_manager &);

	// Static methods
	static bool loose_match(const node_manager &, const node_manager &);
private:
	// TODO: take in node as a reference (const)
	Token *value(Engine *, node) const;

	void unpack(node &);

	size_t count_up(node &);
	
	void label(node &);
	void label_operation(node &);
	
	void rereference(node &);

	node expand(Engine *, const std::string &, const std::vector <node> &,
			const std::set <std::string> & = std::set <std::string> ());

	void simplify(Engine *, node &);
	void simplify_separable(Engine *, node &);
	void simplify_mult_div(Engine *, node &, codes);

	// TODO: Input an engine?
	void differentiate(node &);
	void differentiate_mul(node &);
	void differentiate_pow(node &);
	void differentiate_ln(node &);
	void differentiate_lg(node &);
	void differentiate_const_log(node &);
	void differentiate_trig(node &);
	void differentiate_hyp(node &);
	
	void refactor_reference(node &, const std::string &, Token *);

	std::string display(node) const;
	std::string display_operation(node) const;
	std::string display_pemdas(node, node) const;

	/*
	 * Node factories; produce special nodes such as ones, zeros,
	 * etc. to make constuction of such nodes easy.
	 */
	static node nf_one();
	static node nf_zero();
	
	// General error
	class error {
		std::string str;
	public:
		explicit error(const std::string &s) : str(s) {}

		const std::string &what() const {
			return str;
		}
	};
public:
	// Syntax error
	class syntax_error : public error {
	public:
		explicit syntax_error(const std::string &s)
				: error(s) {}
	};

	// Undefined symbol error
	class undefined_symbol : public error {
	public:
		explicit undefined_symbol(const std::string &s)
				: error(s) {}
	};

	// Static variables
	
	// Use for computation specifically
	static Engine *shared_context;
};

}

#endif
