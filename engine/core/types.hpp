#ifndef doubleYPES_H_
#define doubleYPES_H_

// C/C++ headers
#include <typeindex>

// Engine headers
#include <operand.hpp>
#include <rational.hpp>
#include <complex.hpp>
#include <matrix.hpp>
#include <vector.hpp>

namespace zhetapi {
	
typedef int Z;					
typedef double R;				
typedef bool B;					
typedef std::string S;				
typedef Rational <Z> Q;				
						
typedef Complex <Z> CZ;				
typedef Complex <Q> CQ;				
typedef Complex <R> CR;				
						
typedef Vector <Z> VZ;				
typedef Vector <Q> VQ;				
typedef Vector <R> VR;				
						
typedef Vector <CZ> VCZ;			
typedef Vector <CQ> VCQ;			
typedef Vector <CR> VCR;			
						
typedef Matrix <Z> MZ;				
typedef Matrix <Q> MQ;				
typedef Matrix <R> MR;				
						
typedef Matrix <CZ> MCZ;			
typedef Matrix <CQ> MCQ;			
typedef Matrix <CR> MCR;			
						
typedef Operand <Z> opd_z;      	
typedef Operand <Q> opd_q;      	
typedef Operand <R> opd_r;      	
	       			
typedef Operand <CZ> opd_cz;		
typedef Operand <CQ> opd_cq;		
typedef Operand <CR> opd_cr;		
	       			
typedef Operand <VZ> opd_v_z;		
typedef Operand <VQ> opd_v_q;		
typedef Operand <VR> opd_v_r;		
	       			
typedef Operand <VCZ> opd_v_cz;	
typedef Operand <VCQ> opd_v_cq;	
typedef Operand <VCR> opd_v_cr;	
	       			
typedef Operand <MZ> opd_m_z;		
typedef Operand <MQ> opd_m_q;		
typedef Operand <MR> opd_m_r;		
	       			
typedef Operand <MCZ> opd_m_cz;	
typedef Operand <MCQ> opd_m_cq;	
typedef Operand <MCR> opd_m_cr;	
	       			
typedef Operand <B> opd_b;		
typedef Operand <S> opd_s;

#define	ONE	1


struct types {
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

	static std::string symbol(const ::std::type_index &);
	static std::string proper_symbol(const ::std::type_index &);
};

}

#endif
