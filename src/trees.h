#ifndef TREE_H
#define TREE_H

#include <bits/c++config.h>
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
		ttwrapper(operand <data_t>);
		ttwrapper(operation <operand <data_t>>);
                ttwrapper(const ttwrapper <data_t> &);

		operand <data_t> *get_oper() const;
		operation <operand <data_t>> *get_opn() const;

		bool operator==(operand <data_t>);
		bool operator==(operation <operand <data_t>>);

                bool operator==(const ttwrapper <data_t> &);

		template <typename T>
		friend std::ostream &operator<<(std::ostream &, const ttwrapper <T> &);

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
        ttwrapper <data_t> ::ttwrapper(const ttwrapper <data_t> &ttw)
        {
                t = ttw.t;
                switch (t) {
                case token::OPERAND:
                        oper_t = new operand <data_t> (*ttw.oper_t);
                        opn_t = nullptr;
                        break;
                case token::OPERATION:
                        opn_t = new operation <operand <data_t>> (*ttw.opn_t);
                        oper_t = nullptr;
                        break;
                }
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

	template <typename data_t>
	std::ostream &operator<<(std::ostream &os, const ttwrapper <data_t> &ttw)
        {
                switch (ttw.t) {
                case token::OPERAND:
                        os << *(ttw.oper_t);
                        break;
                case token::OPERATION:
                        os << *(ttw.opn_t);
                        break;
                default:
                        os << "Undefined Wrapper Object";
                        break;
                }

                return os;
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

                std::size_t size() const;
		std::size_t get_index(data_t *) const;

		list *operator()(data_t *);

                list *get(std::size_t);
		list *operator[](std::size_t);
	};

        template <typename data_t>
        std::size_t list <data_t> ::size() const
        {
                auto *cpy = this;
                std::size_t counter = 0;

                while (cpy != nullptr) {
                        std::cout << "#" << counter + 1 << " @";
                        std::cout << cpy << std::endl;
                        cpy = cpy->next;
                        std::cout << "[" << counter << "] @";
                        std::cout << (this)[counter] << std::endl;
                        counter++;
                }

                return counter;
        }
	
	template <typename data_t>
	std::size_t list <data_t> ::get_index(data_t *nd) const
        {
		auto *cpy = this;
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
		list <data_t> *cpy = this;

		while (cpy != nullptr) {
			if (*(cpy->curr->tptr) == *(nd->tptr))
				return cpy;
			cpy = cpy->next;
		}

		return nullptr;
	}

        tempplate <typname data_t>
        

	template <typename data_t>
	list <data_t> *list <data_t> ::operator[](std::size_t i) {
		list <data_t> *cpy = this;
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

		void set_cursor(ttwrapper <data_t> *);
		void set_cursor(const ttwrapper <data_t> &);

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
		root = new node <ttwrapper <data_t> *>;
		
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
	void token_tree <data_t> ::set_cursor(ttwrapper <data_t> *ttwptr)
	{
		node <ttwrapper <data_t> *> *nd = cursor;

		if (ttwptr == nullptr) {
			//std::cout << "nulltpr passed" << std::endl;
			return;
		}

		if (nd == nullptr) {
			cursor = new node <ttwrapper <data_t> *>;
			cursor->dptr = new ttwrapper <data_t> (*ttwptr);
			cursor->parent = nullptr;
			cursor->leaves = nullptr;
                        root = cursor;
		} else {
			cursor->dptr = ttwptr;
			cursor->parent = nd->parent;
			cursor->leaves = nd->leaves;
			delete nd;
		}
	}

	template <typename data_t>
	void token_tree <data_t> ::set_cursor(const ttwrapper <data_t> &ttw)
	{
		set_cursor(&ttw);
	}

	template <typename data_t>
	void token_tree <data_t> ::add_branch(ttwrapper <data_t> *tptr)
	{
		node <ttwrapper <data_t> *> *new_node = new node <ttwrapper <data_t> *>;
		new_node->dptr = new ttwrapper <data_t> (*tptr);
		new_node->leaves = nullptr;
		
		if (cursor == nullptr) { // Throw a null_cursor_exception later
			std::cout << "Cursor is null, reset and come back?";
			std::cout << std::endl;
			return;
		}

		list <node <ttwrapper <data_t> *>> *cleaves = cursor->leaves;

		if (cleaves == nullptr) {
			cleaves = new list <node <ttwrapper <data_t> *>>;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			cleaves->curr->parent = cursor;
                        cursor->leaves = cleaves;
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
		token *tptr = new ttwrapper <data_t> (tok);
		add_branch(tptr);
	}

	template <typename data_t>
	void token_tree <data_t> ::move_left()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			//std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
		cursor = ((cursor->parent->leaves)[index - 1]).curr;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_right()
	{
                std::cout << "in move right function " << std::endl;
		int index;

		if (cursor == nullptr) {
			std::cout << "cursor is null, exiting from move right" << std::endl;
			return;
		}

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent->leaves)->get_index(cursor);
                std::cout << "index: " << index << std::endl;
                IC(cursor->parent->leaves->size());
		cursor = ((cursor->parent->leaves))[index + 1].curr;
	}
	
	template <typename data_t>
	void token_tree <data_t> ::move_up()
	{
		if (cursor == nullptr) {
			//std::cout << "cursor is null, exiting from move up" << std::endl;
			return;
		}
		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			//std::cout << "Cursor has no parent" << std::endl;
			return;
		}

		cursor = cursor->parent;
	}

	template <typename data_t>
	void token_tree <data_t> ::move_down(std::size_t i)
	{
		if (cursor == nullptr) {
			std::cout << "cursor is null, exiting" << std::endl;
			return;
		}

		if (cursor->leaves == nullptr) { // Thrown null_leaves_exception
			//std::cout << "Cursor has no leaves" << std::endl;
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
                std::cout << std::endl << "-------------" << std::endl;
                std::cout << "PRINTING TREE" << std::endl;
		print(root, 1, 0);
                std::cout << std::endl << "ADDRESSES" << std::endl;
                if (cursor != nullptr)
                        IC(cursor->dptr);
                if (root != nullptr)
                        IC(root->dptr);
                std::cout << "-------------" << std::endl;
	}

	template <typename data_t>
	void token_tree <data_t> ::print(node <ttwrapper <data_t> *> *nd,
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

		std::cout << "#" << num << " - " << *(nd->dptr) << std::endl;

		list <node <ttwrapper <data_t> *>> *rleaves = nd->leaves;

		counter = 0;
		while (rleaves != nullptr) {
			print(rleaves->curr, counter + 1, lev + 1);
                        rleaves = rleaves->next;
			counter++;
		}
	}
}

#endif