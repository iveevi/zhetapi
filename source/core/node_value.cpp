#include "../../engine/core/common.hpp"
#include "../../engine/engine.hpp"
#include "../../engine/core/node_manager.hpp"
#include <stdexcept>

namespace zhetapi {

static Targs proper_args(Engine *ctx, const node &tree)
{
	Targs targs;

	Token *tptr;
	for (const node &leaf : tree) {
		tptr = node_value(ctx, leaf);

		if (!is_blank_token(tptr))
			targs.push_back(tptr);
	}

	return targs;
}

static void substitute(Engine *ctx, node &tree)
{
	node_reference *ndrptr;
	if ((ndrptr = tree.cast <node_reference> ())) {
		if (ndrptr->get())
			tree.retokenize(node_value(ctx, *(ndrptr->get())));
	}

	for (node &child : tree)
		substitute(ctx, child);
}

static Token *assignment_node(Engine *ctx, const node &tree)
{	
	// Evaluate first node
	Token *tmp = nullptr;
	// Assign for the other nodes

	// Add index operator for nodes
	size_t nleaves = tree.child_count(); // Use a method instead

	for (size_t i = 1; i < nleaves; i++) {
		// TODO: change to general case where child count is not equal to p count
		if (tree[i].child_count() && tree[i].label() == l_lvalue) {
			node body = tree[0];
			substitute(ctx, body);

			Args args = (tree[i][0].cast <Operand <Args>> ())->get();

			node_manager fbody(body);
			fbody.add_args(args);

			// TODO: check for null lvalue
			tmp = new Function((tree[i].cast <lvalue> ())->symbol(),
				args, fbody);
		} else {
			// TODO: stop recomputing
			tmp = node_value(ctx, tree[0]);
		}

		// Ensure that the node has type Assignable (TODO: merge with lvalue)
		lvalue *lv = tree[i].cast <lvalue> ();
		if (lv) {
			lv->assign(tmp, ctx);

			continue;
		}

		Token *tptr = node_value(ctx, tree[i], true);
		Assignable *asgn = dynamic_cast <Assignable *> (tptr);

		if (asgn)
			asgn->assign(tmp);
		else
			throw std::runtime_error("FIXME: Need an lvalue or Assignable on the left side of an \'=\'");
	}

	return tmp;
}

static Token *branch_node(Engine *context, const node &tree)
{
	Token *output = nullptr;

	// Push new stack
	context = push_and_ret_stack(context);

	// TODO: Make static or something (as an operand)
	Token *true_token = new Operand <bool> (true);
	Token *break_token = new Operand <Token *> ((Token *) 0x1);
	Token *continue_token = new Operand <Token *> ((Token *) 0x2);

	// TODO: Check returns
	for (const node &nd : tree) {
		if (nd.label() == l_else_branch) {
			Token *tptr = node_sequential_value(context, nd[0]);

			// TODO: keep in another function
			if (tptr && tokcmp(break_token, tptr))
				output =  tptr;
			if (tptr && tokcmp(continue_token, tptr))
				output = tptr;
			if (dynamic_cast <Operand <Token *> *> (tptr))
				output = tptr;

			break;
		}

		// Fallthrough if not else
		node predicate = nd[0];

		Token *eval = node_value(context, predicate);
		if (tokcmp(eval, true_token)) {
			Token *tptr = node_sequential_value(context, nd[1]);

			if (tptr && tokcmp(break_token, tptr))
				output =  tptr;
			if (tptr && tokcmp(continue_token, tptr))
				output = tptr;
			if (dynamic_cast <Operand <Token *> *> (tptr))
				output = tptr;

			break;
		}
	}

	// Pop the stack
	context = pop_and_del_stack(context);

	// Keep special returns in mind
	return output;
}

static Token *while_node(Engine *context, const node &tree)
{
	Token *output = nullptr;

	// Push new stack
	context = push_and_ret_stack(context);

	// Make static or something (as an operand)
	Token *true_token = new Operand <bool> (true);

	// NOTE: New struct (types) - All operand <Token *> from 0->10 are special (0 = null, 1 = break, etc)
	Token *break_token = new Operand <Token *> ((Token *) 0x1);
	Token *continue_token = new Operand <Token *> ((Token *) 0x2);

	node predicate = tree[0];
	while (true) {
		Token *eval = node_value(context, predicate);

		// TODO: Check returns (and for null)
		if (tokcmp(eval, true_token)) {
			eval = node_sequential_value(context, tree[1]);

			if (eval && tokcmp(eval, break_token))
				break;
			else if (eval && tokcmp(eval, continue_token))
				continue;
			else if (dynamic_cast <Operand <Token *> *> (eval)) {
				output = eval;

				break;
			}
		} else {
			break;
		}
	}

	// Pop the stack
	context = pop_and_del_stack(context);

	// Keep special outputs in mind
	return output;
}

static Token *for_node(Engine *context, const node &tree)
{
	Token *output = nullptr;

	// Push new stack
	context = push_and_ret_stack(context);

	// TODO: replace with special
	Token *break_token = new Operand <Token *> ((Token *) 0x1);
	Token *continue_token = new Operand <Token *> ((Token *) 0x2);

	node lin = tree[0];
	node body = tree[1];

	lvalue *lv = lin[0].cast <lvalue> ();

	Token *expr = node_value(context, lin[1]);
	Generator *gen = dynamic_cast <Generator *> (expr);
	// if (!lv) throw

	if (!gen)
		throw std::runtime_error("Can only for-loop with generators, trying to loop through <" + std::string(typeid(*expr).name()) + ">");

	Iterator *itr = gen->begin();
	while (itr) {
		lv->assign(itr->value(), context);

		Token *eval = node_sequential_value(context, body);

		// TODO: use the static comparator
		if (eval && tokcmp(eval, break_token))
				break;
		else if (eval && tokcmp(eval, continue_token))
			continue;
		else if (dynamic_cast <Operand <Token *> *> (eval)) {
			output = eval;

			break;
		}

		itr = gen->next(itr);
	}

	// Pop the stack
	context = pop_and_del_stack(context);

	return output;
}

static Token *set_in_node(Engine *ctx, const node &tree)
{
	// For now assume that the first node is an lvalue
	rvalue *rv = tree[0].cast <rvalue> ();

	Set *set = dynamic_cast <Set *> (node_value(ctx, tree[1]));

	// if (!rv) throw
	// if (!set) throw
	
	// TODO: use token factories to return
	if (set->present(rv->get(ctx)))
		return new OpB(true);

	return new OpB(false);
}

static Token *node_null_value(Engine *context, const node &tree)
{
	Token *output = nullptr;
	switch (tree.label()) {
	case l_assignment_chain:
		return assignment_node(context, tree);
	case l_branch:
		return branch_node(context, tree);
	case l_while_loop:
		return while_node(context, tree);
	case l_for_loop:
		return for_node(context, tree);
	case l_set_in:
		return set_in_node(context, tree);
	case l_break_loop:
		return new Operand <Token *> ((Token *) 0x1);
	case l_continue_loop:
		// Again, refactor to some externally retrieved token
		// static from node_manager?
		return new Operand <Token *> ((Token *) 0x2);
	case l_return_alg:
		return new Operand <Token *> (node_value(context, tree[0]));
	default:
		throw std::runtime_error("Unknown execution mode \'"
				+ strlabs[tree.label()] + "\'");

		break;
	}

	return nullptr;
}

// TODO: refactor engine to context
// TODO: take const ref
Token *node_value(Engine *ctx, node tree, bool mref)
{
	// TODO: clean up these variables
	Token *tptr;
	Token *vptr;

	// Variable v;

	rvalue *rv;

	algorithm *aptr;

	std::string ident;

	int size;

	node reffed;

	// Check for operation
	if (tree.null())
		return node_null_value(ctx, tree);

	Token::type t = tree.caller();

	// Preliminary check (not functor or operation)
	switch (t) {
	case Token::opd:
	case Token::token_module:
	case Token::token_collection:
		return tree.copy_token();
	case Token::token_rvalue:
		return (tree.cast <rvalue> ())->get(ctx);
	case Token::token_node_list:
		return (tree.cast <node_list> ())->evaluate(ctx);
	case Token::ndr:
		reffed = *((tree.cast <node_reference> ())->get());

		tree.retokenize(node_value(ctx, reffed));

		return node_value(ctx, tree);
	default:
		break;
	}

	// TODO: put this in its own function
	operation_holder *ophptr = tree.cast <operation_holder> ();

	if (ophptr && ophptr->code == attribute) {
		lvalue *lv = tree[1].cast <lvalue> ();

		// TODO: throw on nullptr (TODO: embed the attribute name into
		// the token)
		std::string at = lv->symbol();

		Targs targs = proper_args(ctx, tree[1]);

		Token *callee = node_value(ctx, tree[0]);
		if (!callee)
			throw node_manager::null_attributee();

		/* std::cout << "targs = " << std::endl;
		for (auto t : targs)
			std::cout << "\t" << t->dbg_str() << std::endl; */

		return callee->attr(at, ctx, targs, tree[1].child_count());
	}

	// TODO: put this in its own function
	Token *iptr;
	if (ophptr && ophptr->code == indexing) {
		tptr = node_value(ctx, tree[0]);

		Indexable *idx = dynamic_cast <Indexable *> (tptr);
		if (!idx)
			throw node_manager::bad_indexable(typeid(*tptr));

		iptr = node_value(ctx, tree[1]);

		return idx->index(iptr, mref);
	}

	// else: this func
	Targs values = proper_args(ctx, tree);

	if (t == Token::oph) {
		tptr = ctx->compute((tree.cast <operation_holder> ())->rep, values);

		if (tree.label() == l_post_modifier) {
			rv = tree[0].cast <rvalue> ();

			ctx->put(rv->symbol(), tptr);

			return rv->get(ctx);
		} else if (tree.label() == l_pre_modifier) {
			rv = tree[0].cast <rvalue> ();

			ctx->put(rv->symbol(), tptr);
		}

		return tptr;
	}

	Functor *ftr = tree.cast <Functor> ();
	if (!ftr)
		throw node_manager::bad_token_type();

	// Return the functor itself if there were no arguments
	if (tree.child_count() || values.size()) {
		std::cout << "Functor evaluate:" << std::endl;
		Token *tptr = ftr->evaluate(ctx, values);
		std::cout << "\ttptr = " << tptr << std::endl;
		if (tptr)
			std::cout << "\t\t" << tptr->dbg_str() << std::endl;
		return tptr;
	}

	return tree.copy_token();
}

// TODO: use const tree instead
Token *node_sequential_value(Engine *context, node tree)
{
	// Assumes that the top node is a sequential
	// TODO: dont make these static pointers (stupid)
	static Token *break_token = new Operand <Token *> ((Token *) 0x1);
	static Token *continue_token = new Operand <Token *> ((Token *) 0x2);

	Token *tptr;
	for (node nd : tree) {
		tptr = node_value(context, nd);

		// Check value for special cases (early returns)
		// TODO: helper func
		if (tptr && tokcmp(break_token, tptr))
			return tptr;
		if (tptr && tokcmp(continue_token, tptr))
			return tptr;
		if (dynamic_cast <Operand <Token *> *> (tptr))
			return tptr;
	}

	return nullptr;
}

}
