#include <core/types.hpp>

#include <function.hpp>

namespace zhetapi {

// Factories
Token *types::one()
{
	return new OpZ(int (ONE));
}

/*
template <class A>
Token *types::convert(A x)
{
	// TODO: is this even useful?
	if (typeid(A) == typeid(Z))
		return new OpZ((Z) x);
	if (typeid(A) == typeid(Q))
		return new OpQ((Q) x);
	if (typeid(A) == typeid(R))
		return new OpR((R) x);
	if (typeid(A) == typeid(CZ))
		return new OpCmpZ(CcmpZ(x, true));	// TODO: what is this stupid constructor
	if (typeid(A) == typeid(CQ))
		return new OpCmpQ((CcmpQ) x);
	if (typeid(A) == typeid(CR))
		return new OpCmpR((CmpR) x);
	
	if (typeid(A) == typeid(VecZ))
		return new OpVecZ((VecZ) x);
	if (typeid(A) == typeid(VecQ))
		return new OpVecQ((VecQ) x);
	if (typeid(A) == typeid(VecR))
		return new OpVecR((VecR) x);
	if (typeid(A) == typeid(VecCZ))
		return new OpVecCmpZ((VecCmpZ) x);
	if (typeid(A) == typeid(VecCQ))
		return new OpVecCmpQ((VecCmpQ) x);
	if (typeid(A) == typeid(VecCR))
		return new OpVecCmpR((VecCmpR) x);
	
	if (typeid(A) == typeid(MZ))
		return new opd_m_z((MZ) x);
	if (typeid(A) == typeid(MQ))
		return new opd_m_q((MQ) x);
	if (typeid(A) == typeid(MR))
		return new opd_m_r((MR) x);
	if (typeid(A) == typeid(MCZ))
		return new opd_m_cz((MCZ) x);
	if (typeid(A) == typeid(MCQ))
		return new opd_m_cq((MCQ) (CQ(x, true)));
	if (typeid(A) == typeid(MCR))
		return new opd_m_cr((MCR) x);

	return nullptr;
} */

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
std::string types::symbol(const std::type_index &type)
{
	// Complex numbers
	// TODO: use ID and a hash table
	if (type == typeid(OpZ))
		return "Z";
	if (type == typeid(OpQ))
		return "Q";
	if (type == typeid(OpR))
		return "R";
	if (type == typeid(OpCmpZ))
		return "CZ";
	if (type == typeid(OpCmpQ))
		return "CQ";
	if (type == typeid(OpCmpR))
		return "CR";
	
	// Vectors
	if (type == typeid(OpVecZ))
		return "VZ";
	if (type == typeid(OpVecQ))
		return "VQ";
	if (type == typeid(OpVecR))
		return "VR";
	if (type == typeid(OpVecCmpZ))
		return "VCZ";
	if (type == typeid(OpVecCmpQ))
		return "VCQ";
	if (type == typeid(OpVecCmpR))
		return "VCR";
	
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

	return type.name();
}

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
