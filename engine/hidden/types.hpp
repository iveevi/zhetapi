#ifndef TYPES_H_
#define TYPES_H_

// C/C++ headers
#include <typeindex>

// Engine headers
#include "../engine/operand.hpp"
#include "../engine/rational.hpp"
#include "../engine/complex.hpp"
#include "../engine/matrix.hpp"

#define __TYPEDEFS__						\
		typedef U Z;					\
		typedef Rational <Z> Q;				\
		typedef T R;					\
								\
		typedef Complex <Z> CZ;				\
		typedef Complex <Q> CQ;				\
		typedef Complex <R> CR;				\
								\
		typedef Vector <Z> VZ;				\
		typedef Vector <Q> VQ;				\
		typedef Vector <R> VR;				\
								\
		typedef Vector <CZ> VCZ;			\
		typedef Vector <CQ> VCQ;			\
		typedef Vector <CR> VCR;			\
								\
		typedef Matrix <Z> MZ;				\
		typedef Matrix <Q> MQ;				\
		typedef Matrix <R> MR;				\
								\
		typedef Matrix <CZ> MCZ;			\
		typedef Matrix <CQ> MCQ;			\
		typedef Matrix <CR> MCR;			\
								\
		typedef zhetapi::Operand <Z> opd_z;      	\
		typedef zhetapi::Operand <Q> opd_q;      	\
		typedef zhetapi::Operand <R> opd_r;      	\
								\
		typedef zhetapi::Operand <CZ> opd_cz;		\
		typedef zhetapi::Operand <CQ> opd_cq;		\
		typedef zhetapi::Operand <CR> opd_cr;		\
								\
		typedef zhetapi::Operand <VZ> opd_v_z;		\
		typedef zhetapi::Operand <VQ> opd_v_q;		\
		typedef zhetapi::Operand <VR> opd_v_r;		\
								\
		typedef zhetapi::Operand <VCZ> opd_v_cz;	\
		typedef zhetapi::Operand <VCQ> opd_v_cq;	\
		typedef zhetapi::Operand <VCR> opd_v_cr;	\
								\
		typedef zhetapi::Operand <MZ> opd_m_z;		\
		typedef zhetapi::Operand <MQ> opd_m_q;		\
		typedef zhetapi::Operand <MR> opd_m_r;		\
								\
		typedef zhetapi::Operand <MCZ> opd_m_cz;	\
		typedef zhetapi::Operand <MCQ> opd_m_cq;	\
		typedef zhetapi::Operand <MCR> opd_m_cr;

#define	ONE	1

namespace zhetapi {

	/*
	 * TYPES:
	 *
	 * Z - Integer
	 * Q - Rational
	 * R - Real
	 *
	 */

	template <class T, class U>
	class types {
	public:
		__TYPEDEFS__
		
		/////////////////////
		// Token factories
		/////////////////////

		static Token *one();

		template <class A>
		static Token *convert(A);
		
		/////////////////////
		// Token identifiers
		/////////////////////
		
		static bool is_zero(Token *);
		static bool is_one(Token *);

		static std::string symbol(const std::type_index &);
		static std::string proper_symbol(const std::type_index &);
	};

	// Factories
	template <class T, class U>
	Token *types <T, U> ::one()
	{
		return new opd_z(U (ONE));
	}

	template <class T, class U>
	template <class A>
	Token *types <T, U> ::convert(A x)
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
	template <class T, class U>
	bool types <T, U> ::is_zero(Token *tptr)
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

	template <class T, class U>
	bool types <T, U> ::is_one(Token *tptr)
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

	template <class T, class U>
	std::string types <T, U> ::symbol(const std::type_index &type)
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

		return "?";
	}

	template <class T, class U>
	std::string types <T, U> ::proper_symbol(const std::type_index &type)
	{
		// Complex numbers

		// Stick to <double, int> for testing
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
		
		std::cout << "TYPE: " << type.name() << std::endl;
		std::cout << "int: " << typeid(Operand <Z>).name() << std::endl;

		return "?";
	}

}

#endif
