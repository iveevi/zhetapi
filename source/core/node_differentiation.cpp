#include "../../engine/core/node_manager.hpp"
#include "../../engine/core/operation_base.hpp"

namespace zhetapi {

void node_manager::differentiate_mul(node &ref)
{
	node ldif(ref[0]);
	differentiate(ldif);

	node rdif(ref[1]);
	differentiate(rdif);

	node tmp(new operation_holder("+"), l_multiplied, {
		node(new operation_holder("*"), l_multiplied, {
			ldif,
			node(ref[1])
		}),
		node(new operation_holder("*"), l_multiplied, {
			node(ref[0]),
			rdif
		})
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_pow(node &ref)
{
	Token *mul = ref[1].copy_token();
	Token *exp = detail::compute("-", {mul, new OpZ(1)});

	node diffed(ref[0]);
	differentiate(diffed);

	node tmp(new operation_holder("*"), l_multiplied, {
		node(new operation_holder("*"), l_multiplied, {
			node(mul, l_none, {}),
			node(new operation_holder("^"), l_power, {
				node(ref[0]),
				node(exp, l_power, {})
			})
		}),
		diffed
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_ln(node &ref)
{
	node diffed(ref[0]);
	differentiate(diffed);

	node tmp(new operation_holder("/"), l_divided, {
		diffed,
		node(ref[0])
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_lg(node &ref)
{
	node diffed(ref[0]);
	differentiate(diffed);

	node tmp(new operation_holder("/"), l_divided, {
		diffed,
		node(new operation_holder("*"), l_multiplied, {
			node(detail::compute("ln", {new OpZ(2)}), l_none, {}),
			node(ref[0])
		})
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_const_log(node &ref)
{
	node diffed(ref[1]);
	differentiate(diffed);

	node tmp(new operation_holder("/"), l_divided, {
		diffed,
		node(new operation_holder("*"), l_multiplied, {
			node(detail::compute("ln", {
				node_value(shared_context, ref[0])
			}), l_none, {}),
			node(ref[1])
		})
	});

	ref.transfer(tmp);
}

void node_manager::differentiate_trig(node &ref)
{
	node diffed(ref[0]);
	differentiate(diffed);

	node op;

	operation_holder *ophptr = ref.cast <operation_holder> ();
	switch (ophptr->code) {
	case sxn:
		op = node(new operation_holder("cos"), l_trigonometric, {
			node(ref[0])
		});
		break;
	case cxs:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sin"), l_trigonometric, {
				node(ref[0])
			}),
			node(new OpZ(-1), l_constant_integer, {})
		});
		break;
	case txn:
		op = node(new operation_holder("^"), l_power, {
			node(new operation_holder("sec"), l_trigonometric, {
				node(ref[0])
			}),
			node(new OpZ(2), l_constant_integer, {})
		});
		break;
	case sec:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sec"), l_trigonometric, {
				node(ref[0])
			}),
			node(new operation_holder("tan"), l_trigonometric, {
				node(ref[0])
			}),
		});
		break;
	case csc:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("*"), l_trigonometric, {
				node(new operation_holder("csc"), l_trigonometric, {
					node(ref[0])
				}),
				node(new operation_holder("cot"), l_trigonometric, {
					node(ref[0])
				}),
			}),
			node(new OpZ(-1), l_constant_integer, {})
		});
		break;
	case cot:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("^"), l_power, {
				node(new operation_holder("cot"), l_trigonometric, {
					node(ref[0])
				}),
				node(new OpZ(2), l_constant_integer, {})
			}),
			node(new OpZ(-1), l_constant_integer, {})
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
	node diffed(ref[0]);
	differentiate(diffed);

	node op;

	operation_holder *ophptr = ref.cast <operation_holder> ();
	switch (ophptr->code) {
	case snh:
		op = node(new operation_holder("cosh"), l_hyperbolic, {
			node(ref[0])
		});
		break;
	case csh:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sinh"), l_hyperbolic, {
				node(ref[0])
			}),
			node(new OpZ(-1), l_constant_integer, {})
		});
		break;
	case tnh:
		op = node(new operation_holder("^"), l_power, {
			node(new operation_holder("sech"), l_hyperbolic, {
				node(ref[0])
			}),
			node(new OpZ(2), l_constant_integer, {})
		});
		break;
	case sch:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("sech"), l_hyperbolic, {
				node(ref[0])
			}),
			node(new operation_holder("tanh"), l_hyperbolic, {
				node(ref[0])
			}),
		});
		break;
	case cch:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("*"), l_multiplied, {
				node(new operation_holder("csch"), l_hyperbolic, {
					node(ref[0])
				}),
				node(new operation_holder("coth"), l_hyperbolic, {
					node(ref[0])
				}),
			}),
			node(new OpZ(-1), l_constant_integer, {})
		});
		break;
	case cth:
		op = node(new operation_holder("*"), l_multiplied, {
			node(new operation_holder("^"), l_power, {
				node(new operation_holder("coth"), l_hyperbolic, {
					node(ref[0])
				}),
				node(new OpZ(2), l_constant_integer, {})
			}),
			node(new OpZ(-1), l_constant_integer, {})
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
