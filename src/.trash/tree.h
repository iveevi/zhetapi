#ifndef TREE_H_
#define TREE_H_

// C++ Standard Libraries
#include <string>
#include <vector>

// Custom Built Libraries
#include "defaults.h"
#include "debug.h"
#include "token.h"
#include "var_stack.h"

template <typename T>
class parser;

// Beginning of tree class - move parser utilities here
template <typename T>
class tree {
public:
	struct node {
		token *tok;
		std::vector <node *> leaves;
	};
private:
	node *root;
	node *cursor;

	void free(node *);
public:
	// Constructors
	tree();
	
	/* ignore these constructors for now
	explicit tree(ttwrapper <T> *);
	explicit tree(const ttwrapper <T> &);
	*/

	explicit tree(std::vector <token *> &);
	explicit tree(std::string, var_stack <T> = var_stack <T> ());

	// Destructor
	~tree();

	// make private and try to add const passing
	node *build(std::vector <token *> &);

	// implement later

	// tree(const tree &);
	// implement later

	// keep this, but make getval private
	const T &value() const;

	const T &getval(node *) const;

	// keep this
	void print() const;

	// make private/protected
	void print(node *, int, int) const;

	// dummy class, rename
	class incomputable_exception {};
};

// tree implementation
template <typename T>
tree <T> ::tree()
{
	// std::cout << "[DEFAULT CONSTRUCTOR]" << std::endl;
	root = nullptr;
	cursor = nullptr;
}

/*template <typename T>
tree <T> ::tree(ttwrapper <T> *tptr)
{
	root = new node <ttwrapper <T> *>;

	root->parent = nullptr;
	root->val = tptr;
	root->leaves = nullptr;

	cursor = root;
}

template <typename T>
tree <T> ::tree(const ttwrapper <T> &tok)
{
	root = new node <ttwrapper <T> *>;
	
	root->parent = nullptr;
	// Make copy constructor for
	// all tokens
	root->val = new token(tok);
	root->leaves = nullptr;

	cursor = root;
}*/

template <typename T>
tree <T> ::tree(std::vector <token *> &toks)
{
	// std::cout << "[VECTOR CONSTRUCTOR]" << std::endl;
	root = build(toks);
	cursor = root;
	// print();
	// std::cout << "Finished Construction" << std::endl;
}

template <typename T>
tree <T> ::tree(std::string input, var_stack <T> vst)
{
	// std::cout << "[STRING CONSTRUCTOR]" << std::endl;
	std::vector <token *> toks = parser <T>
		::get_tokens(input, vst);
	/* token *t;
	std::cout << "[ABOUT TO CONSTRUCTOR]" << std::endl;
	stl_reveal(t, toks, [](token *t) {
	if (t == nullptr)
		return std::string("nullptr");
	return t->str();
	});

	dp_msg("BUILDING"); */
	*this = tree(toks);
	// std::cout << "---PRINTING---" << std::endl;
	// print();
	// std::cout << "Finished String Construction" << std::endl;
}

template <typename T>
tree <T> ::~tree()
{
	//std::cout << "-----DESTROYING TREE-----" << std::endl;

	// fix later free(root);
}

template <typename T>
void tree <T> ::free(node *nd)
{
	if (nd == nullptr)
		return;

	for (auto it = nd->leaves.begin(); it != nd->leaves.end(); it++)
		free(*it);

	delete nd;
}

// make private
template <typename T>
typename tree <T> ::node *tree <T> ::build(std::vector <token *> &toks)
{
	node *out, *oa, *ob;
	token *tptr, *t, norm, *tmp;
	std::size_t i, save;

	//dp_msg("Entering")
	/* stl_reveal(t, toks, [](token *t) {
		if (t == nullptr)
			return std::string("nullptr");
		return t->str();
	}); */
	
	if (toks.size() == 0)
		return nullptr;

	if (toks.size() == 1)
		return new node {toks[0], {}};

	tptr = &defaults <T> ::opers[defaults <T> ::NOPERS];

	// dp_msg("Printing out toks vector");

	//dp_msg("Entering save loop")

	save = 0;
	for (int i = 0; i < toks.size(); i++) {
		t = toks[i];
		///dp_msg("t:");
		// std::cout << t << std::endl;
		// std::cout << t->str() << std::endl;
		if (t->caller() == token::OPERATION &&
			((operation <operand <T>> *) (t))->get_order() <= 
			((operation <operand <T>> *) (tptr))->get_order())
			tptr = t, save = i;
	}

	//dp_var(save);
	//dp_var(tptr->str());
	// stl_reveal(t, toks, [](token *t) {return t->str();});

	// Perform seperate algorithm for functions
	if (((operation <operand <T>> *) (tptr))->get_order() ==
		operation <operand <T>> ::FUNC_LMAX) {
		// Since functions have highest priority
		// the stage at which they are the leading
		// operation (root of subtree) is when all
		// its nodes (operands) belong to it
		// therefore, we just need to transfer operands
		// to the operation in order of parsing
		
		// (until silent multiplication such as 3sin x)
		// only consider operands after the operation

		// dp_var(toks.size());

		out = new node {tptr, {}};
		for (int i = save + 1; i < toks.size(); i++) {
			// dp_msg("looping")
			// dp_var(toks[i]->str())

			//tmp = new ttwrapper <T> ((operand <T> *) toks[i]);
			out->leaves.push_back(new node {toks[i], {}});
		}

		//print(out, 1, 0);

		// dp_msg("DONE")
	} else { // Operation is otherwise binary
		std::vector <token *> a, b;

		a = std::vector <token *> (toks.begin(), toks.begin() + save);

		// dp_msg("a:")
		// stl_reveal(tptr, a, [](token *t) {return t->str();});

		b = std::vector <token *> (toks.begin() + (save + 1), toks.end());

		// dp_msg("b:")
		// stl_reveal(tptr, b, [](token *t) {return t->str();});

		out = new node {tptr, {}};
		oa = build(a);
		ob = build(b);

		out->leaves.push_back(oa);
		out->leaves.push_back(ob);
	}

	return out;
}

// Non-Member helper function
// for general scope things
template <typename T>
const T &tree <T> ::getval(node *nd) const
{
	std::vector <operand <T>> vals;
	node *output;
	int index;

	// dp_msg("Here");

	output = new node ();
	// Return operand wrapper if operand,
	// returns operand wrapper value if
	// operation
	switch (nd->tok->caller()) {
	case token::OPERAND:
		// make notation more convenient
		return (dynamic_cast <operand <T> *> (nd->tok))->get();
	case token::OPERATION:
		// Gather all the operands
		// or values (in general) into
		// a vector
		//dp_var(nd->val->get_token()->str())
		//dp_var(nd->leaves->size())
		for (index = 0; index < nd->leaves.size(); index++) {
			// Should throw error if leaf isnt an
			// operation or an operand
			
			//dp_var(nd->leaves->get(index)->curr->val->get_token()->str())
			switch (nd->leaves[index]->tok->caller()) {
			case token::OPERAND:
				// dp_msg("operand")
				vals.push_back(*(dynamic_cast <operand <T> *> (nd->leaves[index]->tok)));
				break;
			case token::OPERATION:
				// dp_msg("operation")
				// Return value if leaf is an operation
				// is always an operand
				vals.push_back(operand <T> (getval(nd->leaves[index])));
				break;
			}
		}

		// dp_var(nd->val->get_opn()->compute(vals))

		//dp_msg("returning valid");
		return (dynamic_cast <operation <operand <T>> *> (nd->tok))->compute(vals).get();
	}

	//dp_msg("returning invalid");
	// should throw instead
	throw incomputable_exception();
}

template <typename T>
const T &tree <T> ::value() const
{
	// Return operand wrapper if operand,
	// returns operand wrapper value if
	// operation
	return getval(root);
}

template <typename T>
void tree <T> ::print() const
{
	//std::cout << std::endl << "-------------" << std::endl;
	std::cout << "PRINTING TREE" << std::endl;
	print(root, 1, 0);
	//std::cout << std::endl << "ADDRESSES" << std::endl;
	//std::cout << "root @" << root << std::endl;
	//std::cout << "cursor @" << cursor << std::endl;
	//if (cursor != nullptr)
	//        IC(cursor->val);
	//if (root != nullptr)
	//        IC(root->val);
	//std::cout << "-------------" << std::endl;
}

template <typename T>
void tree <T> ::print(node *nd,
		int num, int lev) const
{
	//std::cout << "Inside the print function, num = " << num;
	//std::cout << " and lev = " << lev << std::endl;
	if (nd == nullptr) 
		return;

	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	std::cout << "#" << num << " - " << nd << std::endl;//nd->val->get_token()->str() << std::endl;
	//std::cout << " @" << nd << std::endl;

	counter = 0;
	for (auto it = nd->leaves.begin(); it != nd->leaves.end(); it++, counter++)
		print(*it, counter + 1, lev + 1);
}

#endif
