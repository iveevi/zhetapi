#include <engine.hpp>

#include <core/common.hpp>
#include <core/node_manager.hpp>

namespace zhetapi {

static Token *assignment_node(Engine *context, const node &tree)
{
	// Evaluate first node
	Token *tmp = node_value(context, tree[0]);

	// Assign for the other nodes

	// Add index operator for nodes
	size_t nleaves = tree.child_count(); // Use a method instead

	for (size_t i = 1; i < nleaves; i++) {
		// Ensure that the node has type lvalue
		if (tree[i].label() != l_lvalue)
			throw std::runtime_error("Need an lvalue on the left side of an \'=\'");
		
		lvalue *lv = tree[i].cast <lvalue> ();

		lv->assign(tmp, context);
	}

	return nullptr;
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

	// NOTE: New struct (types) - All operand <Token *> from 0->10 are special (0 = null, 1 = break, etc)
	Token *break_token = new Operand <Token *> ((Token *) 0x1);
	Token *continue_token = new Operand <Token *> ((Token *) 0x2);

	node lin = tree[0];
	node body = tree[1];

	lvalue *lv = lin[0].cast <lvalue> ();
	Generator *gen = dynamic_cast <Generator *>
			(node_value(context, lin[1]));
	// if (!lv) throw
	// if (!gen) throw

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

Token *node_value(Engine *context, node tree)
{
	std::vector <Token *> values;

	Token *tptr;
	Token *vptr;

	// Variable v;

	rvalue *rv;

	algorithm *aptr;

	std::string ident;

	int size;

	node reffed;

	if (tree.null())
		return node_null_value(context, tree);

	operation_holder *ophptr = tree.cast <operation_holder> ();

	if (ophptr && ophptr->code == attribute) {
		lvalue *lv = tree[1].cast <lvalue> ();

		// TODO: throw on nullptr
		std::string at = lv->symbol();

		std::vector <Token *> args;

		for (node leaf : tree[1])
			args.push_back(node_value(context, leaf));

		Token *callee = node_value(context, tree[0]);

		return callee->attr(at, args);
	}

	// else: this func

	// TODO: replace this system (use functor and such and use
	// labels for others like node_list, etc)	
	switch (tree.caller()) {
	case Token::opd:
		return tree.copy_token();
	case Token::token_collection:
		return tree.copy_token();
	case Token::oph:	
		// size = tree._leaves.size();

		for (node leaf : tree)
			values.push_back(node_value(context, leaf));

		tptr = context->compute((tree.cast <operation_holder> ())->rep, values);
		
		if (tree.label() == l_post_modifier) {
			rv = tree[0].cast <rvalue> ();
			
			context->put(rv->symbol(), tptr);

			return rv->get(context)->copy();
		} else if (tree.label() == l_pre_modifier) {
			rv= tree[0].cast <rvalue> ();

			context->put(rv->symbol(), tptr);
		}

		return tptr->copy();
	case Token::token_rvalue:
		return (tree.cast <rvalue> ())->get(context)->copy();
	case Token::ndr:
		reffed = *((tree.cast <node_reference> ())->get());

		tree.retokenize(node_value(context, reffed));

		return node_value(context, tree);
	case Token::token_node_list:
		return (tree.cast <node_list> ())->evaluate(context);
	case Token::ftn:
		if (tree.empty())
			return tree.copy_token();
		
		for (node leaf : tree)
			values.push_back(node_value(context, leaf));

		// TODO: shorten (cast)
		tptr = (tree.cast <Function> ())->compute(values, context);

		return tptr->copy();
	case Token::reg:
		for (node leaf : tree)
			values.push_back(node_value(context, leaf));

		// TODO: add compute()
		tptr = (*(tree.cast <Registrable> ()))(values);

		if (tptr)
			return tptr->copy();

		break;
	case Token::alg:
		for (node leaf : tree)
			values.push_back(node_value(context, leaf));
		
		aptr = tree.cast <algorithm> ();
		tptr = aptr->execute(context, values);

		if (tptr)
			return tptr->copy();

		break;
	default:
		break;
	}

	return nullptr;
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
