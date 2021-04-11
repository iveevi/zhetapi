#include <core/types.hpp>

#include <function.hpp>

namespace zhetapi {

// Factories
Token *types::one()
{
	return new opd_z(int (ONE));
}

template <class A>
Token *types::convert(A x)
{
	if (typeid(A) == typeid(Z))
		return new opd_z((Z) x);
	if (typeid(A) == typeid(Q))
		return new opd_q((Q) x);
	if (typeid(A) == typeid(R))
		return new opd_r((R) x);
	if (typeid(A) == typeid(CZ))
		return new opd_cz(CZ(x, true));
	if (typeid(A) == typeid(CQ))
		return new opd_cq((CQ) x);
	if (typeid(A) == typeid(CR))
		return new opd_cr((CR) x);
	
	if (typeid(A) == typeid(VZ))
		return new opd_v_z((VZ) x);
	if (typeid(A) == typeid(VQ))
		return new opd_v_q((VQ) x);
	if (typeid(A) == typeid(VR))
		return new opd_v_r((VR) x);
	if (typeid(A) == typeid(VCZ))
		return new opd_v_cz((VCZ) x);
	if (typeid(A) == typeid(VCQ))
		return new opd_v_cq((VCQ) x);
	if (typeid(A) == typeid(VCR))
		return new opd_v_cr((VCR) x);
	
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
}

// Identifiers
bool types::is_zero(Token *tptr)
{
	opd_z *_oz = dynamic_cast <opd_z *> (tptr);
	if (_oz != nullptr)
		return (_oz->get() == 0);
	
	opd_q *_oq = dynamic_cast <opd_q *> (tptr);
	if (_oq != nullptr)
		return (_oq->get() == 0);
	
	opd_r *_or = dynamic_cast <opd_r *> (tptr);
	if (_or != nullptr)
		return (_or->get() == 0);

	return false;
}

bool types::is_one(Token *tptr)
{
	opd_z *_oz = dynamic_cast <opd_z *> (tptr);
	if (_oz != nullptr)
		return (_oz->get() == 1);
	
	opd_q *_oq = dynamic_cast <opd_q *> (tptr);
	if (_oq != nullptr)
		return (_oq->get() == 1);
	
	opd_r *_or = dynamic_cast <opd_r *> (tptr);
	if (_or != nullptr)
		return (_or->get() == 1);

	return false;
}

std::string types::symbol(const std::type_index &type)
{
	// Complex numbers
	if (type == typeid(Operand <Z>))
		return "Z";
	if (type == typeid(Operand <Q>))
		return "Q";
	if (type == typeid(Operand <R>))
		return "R";
	if (type == typeid(Operand <CZ>))
		return "CZ";
	if (type == typeid(Operand <CQ>))
		return "CQ";
	if (type == typeid(Operand <CR>))
		return "CR";
	
	// Vectors
	if (type == typeid(Operand <VZ>))
		return "VZ";
	if (type == typeid(Operand <VQ>))
		return "VQ";
	if (type == typeid(Operand <VR>))
		return "VR";
	if (type == typeid(Operand <VCZ>))
		return "VCZ";
	if (type == typeid(Operand <VCQ>))
		return "VCQ";
	if (type == typeid(Operand <VCR>))
		return "VCR";
	
	// Matrices
	if (type == typeid(Operand <MZ>))
		return "MZ";
	if (type == typeid(Operand <MQ>))
		return "MQ";
	if (type == typeid(Operand <MR>))
		return "MR";
	if (type == typeid(Operand <MCZ>))
		return "MCZ";
	if (type == typeid(Operand <MCQ>))
		return "MCQ";
	if (type == typeid(Operand <MCR>))
		return "MCR";

	return type.name();
}

std::string type_name(const std::type_index &type)
{
	// Complex numbers

	// Stick to for testing
	if (type == typeid(Operand <Z>))
		return "int";
	if (type == typeid(Operand <Q>))
		return "Rational <int>";
	if (type == typeid(Operand <R>))
		return "double";
	if (type == typeid(Operand <CZ>))
		return "Complex <int>";
	if (type == typeid(Operand <CQ>))
		return "Complex <Rational <int>>";
	if (type == typeid(Operand <CR>))
		return "Complex <double>";
	
	// Vectors
	if (type == typeid(Operand <VZ>))
		return "Vector <int>";
	if (type == typeid(Operand <VQ>))
		return "Vector <Rational <int>>";
	if (type == typeid(Operand <VR>))
		return "Vector <double>";
	if (type == typeid(Operand <VCZ>))
		return "Vector <Complex <int>>";
	if (type == typeid(Operand <VCQ>))
		return "Vector <Complex <Rational <int>>>";
	if (type == typeid(Operand <VCR>))
		return "Vector <Complex <Rational <int>>>";
	
	// Matrices
	if (type == typeid(Operand <MZ>))
		return "Matrix <int>";
	if (type == typeid(Operand <MQ>))
		return "Matrix <Rational <int>>";
	if (type == typeid(Operand <MR>))
		return "Matrix <double>";
	if (type == typeid(Operand <MCZ>))
		return "Matrix <Complex <int>>";
	if (type == typeid(Operand <MCQ>))
		return "Matrix <Complex <Rational <int>>>";
	if (type == typeid(Operand <MCR>))
		return "Matrix <Complex <double>>";
	
	// Miscelleanous
	if (type == typeid(Function))
		return "Function";

	return "?";
}

}
