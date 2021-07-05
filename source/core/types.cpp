#include "../../engine/function.hpp"
#include "../../engine/core/types.hpp"

namespace zhetapi {

// Factories
Token *types::one()
{
	return new OpZ(int (ONE));
}

// Identifiers
bool types::is_zero(Token *tptr)
{
	OpZ *_oz = dynamic_cast <OpZ *> (tptr);
	if (_oz != nullptr)
		return (_oz->get() == 0);
	
	OpQ *_oq = dynamic_cast <OpQ *> (tptr);
	if (_oq != nullptr)
		return (_oq->get() == 0);
	
	OpR *_or = dynamic_cast <OpR *> (tptr);
	if (_or != nullptr)
		return (_or->get() == 0);

	return false;
}

bool types::is_one(Token *tptr)
{
	OpZ *_oz = dynamic_cast <OpZ *> (tptr);
	if (_oz != nullptr)
		return (_oz->get() == 1);
	
	OpQ *_oq = dynamic_cast <OpQ *> (tptr);
	if (_oq != nullptr)
		return (_oq->get() == 1);
	
	OpR *_or = dynamic_cast <OpR *> (tptr);
	if (_or != nullptr)
		return (_or->get() == 1);

	return false;
}

// TODO: what is this?
const char *types::symbol(const std::type_index &type)
{
	// Complex numbers
	// TODO: use ID and a hash table
	if (type == typeid(OpZ))
		return "Integer";
	if (type == typeid(OpQ))
		return "Rational";
	if (type == typeid(OpR))
		return "Real";
	if (type == typeid(OpCmpZ))
		return "Z[i]";
	if (type == typeid(OpCmpQ))
		return "Q[i]";
	if (type == typeid(OpCmpR))
		return "R[i]";
	
	// Vectors
	if (type == typeid(OpVecZ))
		return "Z^n";
	if (type == typeid(OpVecQ))
		return "Q^n";
	if (type == typeid(OpVecR))
		return "R^n";
	if (type == typeid(OpVecCmpZ))
		return "Z[i]^n";
	if (type == typeid(OpVecCmpQ))
		return "Q[i]^n";
	if (type == typeid(OpVecCmpR))
		return "R[i]^n";
	
	// Matrices
	if (type == typeid(OpMatZ))
		return "MZ";
	if (type == typeid(OpMatQ))
		return "MQ";
	if (type == typeid(OpMatR))
		return "MR";
	if (type == typeid(OpMatCmpZ))
		return "MCZ";
	if (type == typeid(OpMatCmpQ))
		return "MCQ";
	if (type == typeid(OpMatCmpR))
		return "MCR";

	if (type == typeid(OpB))
		return "Bool";
	if (type == typeid(OpS))
		return "String";

	return type.name();
}

// TODO: is this relevant
std::string type_name(const std::type_index &type)
{
	// Complex numbers
	// TODO: use id and hash table

	// Stick to for testing
	if (type == typeid(OpZ))
		return "Integer";
	if (type == typeid(OpQ))
		return "Rational <int>";
	if (type == typeid(OpR))
		return "Real";
	if (type == typeid(OpCmpZ))
		return "Complex <int>";
	if (type == typeid(OpCmpQ))
		return "Complex <Rational <int>>";
	if (type == typeid(OpCmpR))
		return "Complex <double>";
	
	// Vectors
	if (type == typeid(OpVecZ))
		return "Vector <int>";
	if (type == typeid(OpVecQ))
		return "Vector <Rational <int>>";
	if (type == typeid(OpVecR))
		return "Vector <double>";
	if (type == typeid(OpVecCmpZ))
		return "Vector <Complex <int>>";
	if (type == typeid(OpVecCmpQ))
		return "Vector <Complex <Rational <int>>>";
	if (type == typeid(OpVecCmpR))
		return "Vector <Complex <Rational <int>>>";
	
	// Matrices
	if (type == typeid(OpMatZ))
		return "Matrix <int>";
	if (type == typeid(OpMatQ))
		return "Matrix <Rational <int>>";
	if (type == typeid(OpMatR))
		return "Matrix <double>";
	if (type == typeid(OpMatCmpZ))
		return "Matrix <Complex <int>>";
	if (type == typeid(OpMatCmpQ))
		return "Matrix <Complex <Rational <int>>>";
	if (type == typeid(OpMatCmpR))
		return "Matrix <Complex <double>>";
	
	// Miscelleanous
	if (type == typeid(OpB))
		return "Bool";
	if (type == typeid(OpS))
		return "String";
	if (type == typeid(Function))
		return "Function";

	return type.name();
}

}
