#ifndef TREE_H
#define TREE_H

#include <type_traits>

#include "tokens.h"
#include "debug.h"

namespace trees {
	// Using declarations
	using namespace tokens;

	// Beginning of ttwrapper class
	template <typename data_t>
	class ttwrapper {
		operand <data_t> *oper_t;
		operation <operand <data_t>> *opn_t;
	public:
		ttwrapper();
		explicit ttwrapper(operand <data_t>);
		explicit ttwrapper(operation <operand <data_t>>);

		operand <data_t> *get_oper() const;
		operation <operand <data_t>> *get_opn() const;

		bool operator==(operand <data_t>);
		bool operator==(operation <operand <data_t>>);

                bool operator==(const ttwrapper <data_t> &);

		token::type t;
	};

	template <typename data_t>
	ttwrapper <data_t> ::ttwrapper()
	{
		oper_t = nullptr;
		opn_t = nullptr;
		t = token::NONE;
	}

	template <typename data_t>
	ttwrapper <data_t> ::ttwrapper(operand <data_t> oper)
	{
		oper_t = &oper;
		opn_t = nullptr;
		t = token::OPERAND;
	}

	template <typename data_t>
	ttwrapper <data_t> ::ttwrapper(operation <operand <data_t>> opn)
	{
		opn_t = &opn;
		oper_t = nullptr;
		t = token::OPERATION;
	}

	template <typename data_t>
	operand <data_t> *ttwrapper <data_t> ::get_oper() const
	{
		return oper_t;
	}

	template <typename data_t>
	operation <operand <data_t>> *ttwrapper <data_t> ::get_opn() const
	{
		return opn_t;
	}

	template <typename data_t>
	bool ttwrapper <data_t> ::operator==(operand <data_t> oper)
	{
		// Throw error
		if (t != token::OPERAND)
			return false;
		return *oper_t == oper;
	}

	template <typename data_t>
	bool ttwrapper <data_t> ::operator==(operation <operand <data_t>> opn)
	{
		// Throw error
		if (t != token::OPERATION)
			return false;
		return *opn_t == opn;
	}

        template <typename data_t>
	bool ttwrapper <data_t> ::operator==(const ttwrapper <data_t> &ttw)
	{
		switch (ttw.t) {
                case token::OPERAND:
                        return *this == *(ttw.oper_t);
                case token::OPERATION:
                        return *this == *(ttw.opn_t);
                }

                return false;
	}

	// Beginning of tree class
	class tree {
	public:
		enum type {TREE, TTREE, VTREE,
			FTREE, ETREE};

		virtual type caller();
	};

	tree::type tree::caller()
	{
		return TREE;
	}

	// Beginning of the helper classes
	// such as node and list

	// Beginning of list class
	template <typename data_t>
	struct list {
		data_t *curr;
		list *next;

		std::size_t get_index(data_t *);

		list *operator()(data_t *);
		list *operator[](std::size_t);
	};
	
	template <typename data_t>
	std::size_t list <data_t> ::get_index(data_t *nd) {
		list *cpy = this;
		int index = 0;

		while (cpy != nullptr) {
			if (*(cpy->curr->dptr) == *(nd->dptr))
				return index;
			cpy = cpy->next;
			index ++;
		}

		return - 1;
	}

	template <typename data_t>
	list <data_t> *list <data_t> ::operator()(data_t *nd) {
		list *cpy = this;

		while (cpy != nullptr) {
			if (*(cpy->curr->tptr) == *(nd->tptr))
				return cpy;
			cpy = cpy->next;
		}

		return nullptr;
	}

	template <typename data_t>
	list <data_t> *list <data_t> ::operator[](std::size_t i) {
		list *cpy = this;
		int index = i;

		while (index >= 0) {
			if (cpy == nullptr)
				return nullptr;
			cpy = cpy->next;
			index --;
		}

		return cpy;
	}

	// Beginning of node class
	template <typename data_t>
	struct node {
		data_t dptr;
		node *parent;
		list <node> *leaves;
	};

	// Beginning of token_tree class
	template <typename data_t>
	class token_tree {
		node <ttwrapper <data_t> *> *root;
		node <ttwrapper <data_t> *> *cursor;
	public:
		// Constructors
		token_tree();
		explicit token_tree(ttwrapper <data_t> *);
		explicit token_tree(const ttwrapper <data_t> &);

		// token_tree(std::string);
		// implement later

		// token_tree(const token_tree &);
		// implement later

		void reset_cursor();

		void add_branch(ttwrapper <data_t> *);
		void add_branch(const ttwrapper <data_t> &);

		void move_left();
		void move_right();
		void move_up();
		void move_down(std::size_t);

		node <ttwrapper <data_t> *> *current();

		void print() const;
		void print(node <ttwrapper <data_t> *> *, int, int) const;
	};

	// token_tree implementation
	template <typename data_t>
	token_tree <data_t> ::token_tree()
	{
		root = nullptr;
		cursor = nullptr;
	}

	template <typename data_t>
	token_tree <data_t> ::token_tree(ttwrapper <data_t> *tptr)
	{
		root = new node <ttwrapper <data_t> *>;

		root->parent = nullptr;
		root->dptr = tptr;
		root->leaves = nullptr;

		cursor = root;
	}

	template <typename data_t>
	token_tree <data_t> ::token_tree(const ttwrapper <data_t> &tok)
	{
		root = new node <token *>;
		
		root->parent = nullptr;
		// Make copy constructor for
		// all tokens
		root->dptr = new token(tok);
		root->leaves = nullptr;

		cursor = root;
	}

	template <typename data_t>
	void token_tree <data_t> ::reset_cursor()
	{
		cursor = root;
	}

	template <typename data_t>
	void token_tree <data_t> ::add_branch(ttwrapper <data_t> *tptr)
	{
                extern int stage;
                stage = 

		node <ttwrapper <data_t> *> *new_node = new node <ttwrapper
		        <data_t> *>;
		new_node->dptr = tptr;
		new_node->leaves = nullptr;

                stage = 0x10;

		list <node <ttwrapper <data_t> *>> *cleaves = cursor->leaves;

		if (cursor == nullptr) { // Throw a null_cursor_exception later
			std::cout << "Cursor is null, reset and come back?";
			std::cout << std::endl;
		}

		if (cleaves == nullptr) {
			cleaves = new list <node <ttwrapper <data_t> *>>;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			new_node->parent = cursor;
		} else {
			while (cleaves->next != nullptr)
				cleaves = cleaves->next;
			cleaves->next = new list <node <ttwrapper <data_t> *>>;
			cleaves = cleaves->next;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			new_node->parent = cursor;
		}
	}
	
	template <typename data_t>
	void token_tree <data_t> ::add_branch(const ttwrapper <data_t> &tok)
	{
		token *tptr = new token(tok);
		add_branch(tptr);
	}

	template <typename data_t>
	void token_tree <data_t> ::move_left()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
		cursor = ((cursor->parent->leaves)[index - 1]).curr;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_right()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
		cursor = ((cursor->parent->leaves))[index + 1].curr;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_up()
	{
		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "Cursor has no parent" << std::endl;
			return;
		}

		cursor = cursor->parent;
	}

	template <typename data_t>
	void token_tree <data_t> ::move_down(std::size_t i)
	{
		if (cursor->leaves == nullptr) { // Thrown null_leaves_exception
			std::cout << "Cursor has no leaves" << std::endl;
			return;
		}

		cursor = (cursor->leaves)[i].curr;
	}

	template <typename data_t>
	node <ttwrapper <data_t> *> *token_tree <data_t> ::current()
	{
		return cursor;
	}

	template <typename data_t>
	void token_tree <data_t> ::print() const
	{
		print(root, 1, 0);
	}

	template <typename data_t>
	void token_tree <data_t> ::print(node <ttwrapper <data_t> *> *nd,
			int num, int lev) const
	{
		if (nd == nullptr)
			return;

		int counter = lev;
		while (counter > 0) {
			std::cout << "\t";
			counter--;
		}

		std::cout << "#" << num << " - ";
		
		switch (nd->dptr->t) {
		case token::OPERAND:
			std::cout << *(nd->dptr->get_oper());
			break;
		case token::OPERATION:
			std::cout << *(nd->dptr->get_opn());
			break;
		default:	
			std::cout << "Invalid dptr kind";
			break;
		}

		std::cout << std::endl;

		list <node <ttwrapper <data_t> *>> *rleaves = nd->leaves;

		counter = 0;
		while (rleaves != nullptr) {
			print(rleaves->curr, counter + 1, lev + 1);
			counter++;
		}
	}
}

#endif