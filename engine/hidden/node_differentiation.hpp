#ifndef NODE_DIFFERENTIATION_H_
#define NODE_DIFFERENTIATION_H_

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

	template <class T, class U>
	void node_manager <T, U> ::differentiate_trig(node &ref)
	{
		node diffed(ref.__leaves[0]);
		differentiate(diffed);

		node op;

		operation_holder *ophptr = dynamic_cast <operation_holder *> (ref.__tptr.get());
		switch (ophptr->code) {
		case sin:
			op = node(new operation_holder("cos"), l_trigonometric, {
				node(ref.__leaves[0])
			});
			break;
		case cos:
			op = node(new operation_holder("*"), l_trigonometric, {
				node(new operation_holder("sin"), l_trigonometric, {
					node(ref.__leaves[0])
				}),
				node(new Operand <U> (-1), l_constant_integer, {})
			});
			break;
		case tan:
			op = node(new operation_holder("^"), l_trigonometric, {
				node(new operation_holder("sec"), l_trigonometric, {
					node(ref.__leaves[0])
				}),
				node(new Operand <U> (2), l_constant_integer, {})	
			});
			break;
		case sec:
			op = node(new operation_holder("*"), l_trigonometric, {
				node(new operation_holder("sec"), l_trigonometric, {
					node(ref.__leaves[0])
				}),
				node(new operation_holder("tan"), l_trigonometric, {
					node(ref.__leaves[0])
				}),
			});
			break;
		case csc:
			op = node(new operation_holder("*"), l_trigonometric, {
				node(new operation_holder("*"), l_trigonometric, {
					node(new operation_holder("csc"), l_trigonometric, {
						node(ref.__leaves[0])
					}),
					node(new operation_holder("cot"), l_trigonometric, {
						node(ref.__leaves[0])
					}),
				}),
				node(new Operand <U> (-1), l_constant_integer, {})
			});
			break;
		case cot:
			op = node(new operation_holder("*"), l_trigonometric, {
				node(new operation_holder("^"), l_trigonometric, {
					node(new operation_holder("cot"), l_trigonometric, {
						node(ref.__leaves[0])
					}),
					node(new Operand <U> (2), l_constant_integer, {})	
				}),
				node(new Operand <U> (-1), l_constant_integer, {})
			});
			break;
		default:
			break;
		}

		node tmp(new operation_holder("*"), l_multiplied, {
			diffed,
			op
		});

		ref.transfer(tmp);
	}
	
}

#endif