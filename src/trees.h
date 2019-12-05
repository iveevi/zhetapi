#ifndef TREE_H
#define TREE_H

#include "tokens.h"

namespace trees {
	// Beginning of tree class
	class tree {
	public:
		enum type {TREE, TTREE, VTREE,
			FTREE, ETREE};
		virtual type caller();
	};

	tree::type tree:caller()
	{
		return TREE;
	}

	// Beginning of token_tree class
	class token_tree {
		struct list {
			node *curr;
			list *next;

			std::size_t get_index(node *);

			list *operator()(node *);
			list *operator[](std::size_t);
		};

		struct node {
			node *parent;
			token *tptr;
			list *leaves;
		};

		node *root;
		node *current;
	public:
		token_tree();
		token_tree(token *);
		token_tree(const token &);
		
		// token_tree(std::string);
		// implement later
		
		// token_tree(const token_tree &);
		// implement later
		
		void reset_cursor();
		
		void add_branch(token *);
		void add_branch(const token &);

		void move_left();
		void move_right();
		void move_up();
		void move_down(std::size_t);

		node *current();
		
		void print() const;	
		void print(node *) const;
	}

	// list implementation
	std::size_t token_tree::list::get_index(node *nd)
	{
		list *cpy = this;
		int index = 0;

		while (cpy != nullptr) {
			if (*(cpy->curr->tptr) == *(nd->tptr))
				return index;
			cpy = cpy->next;
			index++;
		}

		return -1;
	}

	list *token_tree::list::operator()(node *nd)
	{
		list *cpy = this;

		while (cpy != nullptr) {
			if (*(cpy->curr->tptr) == *(nd->tptr))
				return cpy;
			cpy = cpy->next;
		}

		return nullptr;
	}

	list *token_tree::list::operator[](std::size_t i)
	{
		list *cpy = this;
		int index = i;

		while (index >= 0) {
			if (cpy == nullptr)
				return nullptr;
			cpy = cpy->next;
		}

		return cpy;
	}

	// token_tree implementation
	token_tree::token_tree()
	{
		root = nullptr;
		cursor = nullptr;
	}

	token_tree::token_tree(token *tptr)
	{
		root = new node;

		root->parent = nullptr;
		root->tptr = tptr;
		root->leaves = nullptr;

		cursor = root;
	}

	token_tree::token_tree(const token &tok)
	{
		root = new node;
		
		root->parent = nullptr;
		root->tptr = &tok;
		root->leaves = nullptr;

		cursor = root;
	}

	void token_tree::reset_cursor()
	{
		cursor = root;
	}

	void token_tree::add_branch(token *tptr)
	{
		node *new_node = new node;
		new_node->tptr = tptr;
		new_node->leaves = nullptr;

		list *cleaves = cursor->leaves;

		if (cursor == nullptr) { // Throw a null_cursor_exception later
			std::cout << "Cursor is null, reset and come back?";
			std::cout << std::endl;
		}

		if (cleaves == nullptr) {
			cleaves = new list;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			new_node->parent = cursor;
		} else {
			while (cleaves->next != nullptr)
				cleaves = cleaves->next;
			cleaves->next = new list;
			cleaves = cleaves->next;
			cleaves->curr = new_node;
			cleaves->next = nullptr;
			new_node->parent = cursor;
		}
	}
	
	void token_tree::add_branch(const token &tok)
	{
		token *tptr = &tok;
		add_branch(tptr);
	}

	void token_tree::move_left()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent)->get_index(cursor);
		cursor = (cursor->parent)[index - 1];
	}
	
	void token_tree::move_left()
	{
		int index;

		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "No nodes beside cursor" << std::endl;
			return;
		}

		index = (cursor->parent)->get_index(cursor);
		cursor = (cursor->parent)[index + 1];
	}
	
	void token_tree::move_up()
	{
		if (cursor->parent == nullptr) { // Thrown null_parent_exception
			std::cout << "Cursor has no parent" << std::endl;
			return;
		}

		cursor = cursor->parent;
	}

	void token_tree::move_down(std::size_t i)
	{
		if (cursor->leaves == nullptr) { // Thrown null_leaves_exception
			std::cout << "Cursor has no leaves" << std::endl;
			return;
		}

		cursor = (cursor->leaves)[i];
	}

	node *token_tree::current()
	{
		return cursor;
	}

	void token_tree::print() const
	{
		print(root);
	}

	void token_tree::print(node *nd) const
	{
		node *rcpy = nd;

		if (rcpy == nullptr)
			return;

		std::cout << *(rcpy->tptr) << " ";
		list *rleaves = rcpy->leaves;

		int counter = 0;
		while (rleaves != nullptr) {
			std::cout << "\t#" << counter;
			print(rleaves->curr);
			counter++;
		}
	}
}

#endif
