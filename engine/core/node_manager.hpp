#ifndef NODE_MANAGER_H_
#define NODE_MANAGER_H_

// C/C++ headers
#include <fstream>
#include <stack>

// Engine headers
#include <core/node_reference.hpp>
#include <core/parser.hpp>
#include <core/types.hpp>

namespace zhetapi {

class Barn;
class Function;

class node_manager {
public:
	__TYPEDEFS__
private:
	Barn *				__barn;
	node				__tree;
	std::vector <node>		__refs;
	std::vector <std::string>	__params;
public:
	node_manager();
	node_manager(const node_manager &);
	node_manager(const std::string &, Barn * = nullptr);
	node_manager(const std::string &, const std::vector <std::string> &, Barn * = nullptr);

	node_manager &operator=(const node_manager &);

	node tree() const;

	Token *value() const;
	Token *value(Barn *) const;

	Token *substitute_and_compute(std::vector <Token *> &, size_t = 1);

	void append(const node &);
	void append(const node_manager &);

	/*
	 * Responsible for expanding variable clusters and truning them
	 * into product of Operands.
	 */
	void expand(node &);

	void simplify();

	void differentiate(const std::string &);

	void refactor_reference(const std::string &, Token *);

	/*
	 * Code generator. Requires the specification of an output file.
	 */
	void generate(std::string &) const;

	std::string display() const;

	void print(bool = false) const;

	// Static methods
	static bool loose_match(const node_manager &, const node_manager &);
private:
	Token *value(node) const;
	Token *value(node, Barn *) const;

	size_t count_up(node &);
	
	void label(node &);
	void label_operation(node &);
	
	void rereference(node &);

	node expand(const std::string &, const std::vector <node> &);

	void simplify(node &);
	void simplify_separable(node &);

	void differentiate(node &);
	void differentiate_mul(node &);
	void differentiate_pow(node &);
	void differentiate_ln(node &);
	void differentiate_lg(node &);
	void differentiate_const_log(node &);
	void differentiate_trig(node &);
	void differentiate_hyp(node &);
	
	void refactor_reference(node &, const std::string &, Token *);

	std::string generate(std::string, node, std::ofstream &, size_t &, size_t &) const;

	std::string display(node) const;
	std::string display_operation(node) const;
	std::string display_pemdas(node, node) const;

	/*
	 * Node factories; produce special nodes such as ones, zeros,
	 * etc. to make constuction of such nodes easy.
	 */
	static node nf_one();
	static node nf_zero();
public:
	// General error
	class error {
		std::string str;
	public:
		error(std::string s) : str(s) {}

		const std::string &what() const {
			return str;
		}
	};

	// Syntax error
	class syntax_error : public error {
	public:
		syntax_error(std::string s) : error(s) {}
	};

	// Undefined symbol error
	class undefined_symbol : public error {
	public:
		undefined_symbol(std::string s) : error(s) {}
	};
};

}

#endif
