#ifndef NODE_DIFFERENTIATION_H_
#define NODE_DIFFERENTIATION_H_

// Engine headers
#include <hidden/node_manager.hpp>

namespace zhetapi {

	template <class T, class U>
	void node_manager <T, U> ::differentiate_mul(node &ref)
	{
		node ldif(ref.__leaves[0]);
		differentiate(ldif);

		node rdif(ref.__leaves[1]);
		differentiate(rdif);

		node tmp(new operation_holder("+"), l_multiplied, {
			node(new operation_holder("*"), l_multiplied, {
				ldif,
				node(ref.__leaves[1])
			}),
			node(new operation_holder("*"), l_multiplied, {
				node(ref.__leaves[0]),
				rdif
			})
		});

		ref.transfer(tmp);
	}

	template <class T, class U>
	void node_manager <T, U> ::differentiate_pow(node &ref)
	{
		Token *mul = ref.__leaves[1].__tptr->copy();
		Token *exp = __barn.compute("-", {mul, new Operand <U> (1)});

		node diffed(ref.__leaves[0]);
		differentiate(diffed);

		node tmp(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("*"), l_multiplied, {
				node(mul, l_none, {}),
				node(new operation_holder("^"), l_power, {
					node(ref.__leaves[0]),
					node(exp, l_power, {})
				})
			}),
			diffed
		});

		ref.transfer(tmp);
	}

	template <class T, class U>
	void node_manager <T, U> ::differentiate_ln(node &ref)
	{
		node diffed(ref.__leaves[0]);
		differentiate(diffed);

		node tmp(new operation_holder("/"), l_divided, {
			diffed,
			node(ref.__leaves[0])
		});

		ref.transfer(tmp);
	}

	template <class T, class U>
	void node_manager <T, U> ::differentiate_lg(node &ref)
	{
		node diffed(ref.__leaves[0]);
		differentiate(diffed);

		node tmp(new operation_holder("/"), l_divided, {
			diffed,
			node(new operation_holder("*"), l_multiplied, {
				node(__barn.compute("ln", {new Operand <U> (2)}), l_none, {}),
				node(ref.__leaves[0])
			})
		});

		ref.transfer(tmp);
	}

	template <class T, class U>
	void node_manager <T, U> ::differentiate_const_log(node &ref)
	{
		node diffed(ref.__leaves[1]);
		differentiate(diffed);

		node tmp(new operation_holder("/"), l_divided, {
			diffed,
			node(new operation_holder("*"), l_multiplied, {
				node(__barn.compute("ln", {value(ref.__leaves[0])}), l_none, {}),
				node(ref.__leaves[1])
			})
		});

		ref.transfer(tmp);
	}
	
}

#endif