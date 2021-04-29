#include <core/node_manager.hpp>
#include <engine.hpp>

namespace zhetapi {

void node_manager::differentiate_mul(node &ref)
{
	node ldif(ref._leaves[0]);
	differentiate(ldif);

	node rdif(ref._leaves[1]);
	differentiate(rdif);

	node tmp(new operation_holder("+"), l_multiplied, {
		node(new operation_holder("*"), l_multiplied, {
			ldif,
			node(ref._leaves[1])
		}),
		node(new operation_holder("*"), l_multiplied, {
			node(ref._leaves[0]),
			rdif
		})
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_pow(node &ref)
{
	Token *mul = ref._leaves[1]._tptr->copy();
	Token *exp = shared_context->compute("-", {mul, new opd_z(1)});

	node diffed(ref._leaves[0]);
	differentiate(diffed);

	node tmp(new operation_holder("*"), l_multiplied, {
		node(new operation_holder("*"), l_multiplied, {
			node(mul, l_none, {}),
			node(new operation_holder("^"), l_power, {
				node(ref._leaves[0]),
				node(exp, l_power, {})
			})
		}),
		diffed
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_ln(node &ref)
{
	node diffed(ref._leaves[0]);
	differentiate(diffed);

	node tmp(new operation_holder("/"), l_divided, {
		diffed,
		node(ref._leaves[0])
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_lg(node &ref)
{
	node diffed(ref._leaves[0]);
	differentiate(diffed);

	node tmp(new operation_holder("/"), l_divided, {
		diffed,
		node(new operation_holder("*"), l_multiplied, {
			node(shared_context->compute("ln", {new opd_z(2)}), l_none, {}),
			node(ref._leaves[0])
		})
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_const_log(node &ref)
{
	node diffed(ref._leaves[1]);
	differentiate(diffed);

	node tmp(new operation_holder("/"), l_divided, {
		diffed,
		node(new operation_holder("*"), l_multiplied, {
			node(shared_context->compute("ln", {
				node_value(shared_context, ref._leaves[0])
			}), l_none, {}),
			node(ref._leaves[1])
		})
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_trig(node &ref)
{
	node diffed(ref._leaves[0]);
	differentiate(diffed);

	node op;

	operation_holder *ophptr = dynamic_cast <operation_holder *> (ref._tptr);
	switch (ophptr->code) {
	case sxn:
		op = node(new operation_holder("cos"), l_trigonometric, {
			node(ref._leaves[0])
		});
		break;
	case cxs:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sin"), l_trigonometric, {
				node(ref._leaves[0])
			}),
			node(new opd_z(-1), l_constant_integer, {})
		});
		break;
	case txn:
		op = node(new operation_holder("^"), l_power, {
			node(new operation_holder("sec"), l_trigonometric, {
				node(ref._leaves[0])
			}),
			node(new opd_z(2), l_constant_integer, {})	
		});
		break;
	case sec:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sec"), l_trigonometric, {
				node(ref._leaves[0])
			}),
			node(new operation_holder("tan"), l_trigonometric, {
				node(ref._leaves[0])
			}),
		});
		break;
	case csc:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("*"), l_trigonometric, {
				node(new operation_holder("csc"), l_trigonometric, {
					node(ref._leaves[0])
				}),
				node(new operation_holder("cot"), l_trigonometric, {
					node(ref._leaves[0])
				}),
			}),
			node(new opd_z(-1), l_constant_integer, {})
		});
		break;
	case cot:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("^"), l_power, {
				node(new operation_holder("cot"), l_trigonometric, {
					node(ref._leaves[0])
				}),
				node(new opd_z(2), l_constant_integer, {})	
			}),
			node(new opd_z(-1), l_constant_integer, {})
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

void node_manager::differentiate_hyp(node &ref)
{
	node diffed(ref._leaves[0]);
	differentiate(diffed);

	node op;

	operation_holder *ophptr = dynamic_cast <operation_holder *> (ref._tptr);
	switch (ophptr->code) {
	case snh:
		op = node(new operation_holder("cosh"), l_hyperbolic, {
			node(ref._leaves[0])
		});
		break;
	case csh:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sinh"), l_hyperbolic, {
				node(ref._leaves[0])
			}),
			node(new opd_z(-1), l_constant_integer, {})
		});
		break;
	case tnh:
		op = node(new operation_holder("^"), l_power, {
			node(new operation_holder("sech"), l_hyperbolic, {
				node(ref._leaves[0])
			}),
			node(new opd_z(2), l_constant_integer, {})	
		});
		break;
	case sch:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sech"), l_hyperbolic, {
				node(ref._leaves[0])
			}),
			node(new operation_holder("tanh"), l_hyperbolic, {
				node(ref._leaves[0])
			}),
		});
		break;
	case cch:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("*"), l_multiplied, {
				node(new operation_holder("csch"), l_hyperbolic, {
					node(ref._leaves[0])
				}),
				node(new operation_holder("coth"), l_hyperbolic, {
					node(ref._leaves[0])
				}),
			}),
			node(new opd_z(-1), l_constant_integer, {})
		});
		break;
	case cth:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("^"), l_power, {
				node(new operation_holder("coth"), l_hyperbolic, {
					node(ref._leaves[0])
				}),
				node(new opd_z(2), l_constant_integer, {})	
			}),
			node(new opd_z(-1), l_constant_integer, {})
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
