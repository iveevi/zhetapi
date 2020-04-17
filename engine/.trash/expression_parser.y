%{
	#include <iostream>
	#include <cctype>
	#include <cstring>
	#include <vector>
	#include <stack>
	#include <string>

	#include "operation.h"
	#include "defaults.h"
  	#include "operand.h"
	
	#include "var_stack.h"
	#include "variable.h"

	extern "C" int yylex();
  	
	using namespace std;

  	template <class T>
	void yyerror (operand <T> *, var_stack <T>, const char *error);
%}

%define api.prefix {e}

%define parse.error verbose

%parse-param {operand <double> *value}
%parse-param {var_stack <double> vst}

%token E_IDENT
%token E_NUMBER

%token E_PLUS
%token E_MINUS
%token E_MULT
%token E_DIV

%token E_SIN E_COS E_TAN
%token E_CSC E_SEC E_COT
%token E_LOG E_LN E_LG

%token E_SUPERSCRIPT
%token E_SUBSCRIPT

%token E_LPAREN E_RPAREN
%token E_LBRACE E_RBRACE
%token E_LBRACKET E_RBRACKET

%token E_END

%union {
	operand <double>		*expr;
	operand <double>		*coll;
	operand <double>		*term;
	operand <double>		*felm;
	operand <double>		*dopn;
	operand <double>		*dpnt;
	operand <double>		*prth;
	operand <double>		*sclr;

	const char			*ident;
	double       			value;
}

/* Types for the terminal symbols */
%type	<value>	E_NUMBER
%type	<ident>	E_IDENT

/* Types for non-terminal symbols */
%type	<expr>	expr
%type	<coll>	coll
%type	<term>	term
%type	<felm>	felm
%type	<dopn>	dopn
%type	<dpnt>	dpnt
%type	<prth>	prth
%type	<sclr>	sclr

/* Precedence information to resolve ambiguity */
%left E_PLUS E_MINUS
%left E_MULT DIV

%precedence E_LBRACKET E_RBRACKET
%precedence E_LBRACE E_RBRACE
%precedence E_LPAREN E_RPAREN

%precedence E_SUPERSCRIPT
%precedence E_SIN E_COS TAN
%precedence E_CSC E_SEC E_COT
%precedence E_LOG E_LN E_LG

%%

/* make computations based to template type later */

/* Input: general user input */
input:	expr E_END {
     		value->set($1->get());
		return 0;
};

/* Expression: general exprression */
expr:  	expr E_SUPERSCRIPT expr { // Exponentiation
   		printf("expression exponentiation\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals));
} %prec E_SUPERSCRIPT

   |	expr E_MULT expr { // Multiplication
   		printf("expression multiplication\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec E_MULT

   |	expr DIV expr { // Division
   		printf("expression divition\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::div_op(vals));
} %prec DIV

   |	expr E_PLUS expr { // Addition
   		printf("expression addition\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::add_op(vals));
} %prec E_PLUS

   |	expr E_MINUS expr { // Subtraction
   		printf("expression substraction\n");
   		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::sub_op(vals));
} %prec E_MINUS

   | 	E_MINUS coll {
   		printf("expression negative collective\n");
   		vector <operand <double>> vals;
		vals.push_back(operand <double> (-1));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec E_MINUS

   |	coll {
   		printf("expression collective\n");
   		$$ = $1;
} %prec E_LOG;

/* Collective: terms and miscellanics */
coll:	term felm { // Implicit Multiplication: term and non-arithmetic operation
    		printf("collective, term (%s) implicitly multiplicied with non-arithmetic operation (%s)\n", $1->str().c_str(), $2->str().c_str());
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec E_LOG

    |	felm {
    		$$ = $1;
} %prec E_LOG

    |	term {
    		printf("collective as a regular term (%s)\n", $1->str().c_str());
    		$$ = $1;
} %prec E_MULT;

/* Term: algebraic term */
term:	term term { // Implicit Multiplication: two or more terms
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec E_MULT
    		
    |	dopn { // Direct Operand
    		$$ = $1;
};

/* Functional Elementary Operations: non-arithmetic operations */
felm:	E_LOG E_SUBSCRIPT E_LBRACE expr E_RBRACE expr {
    		printf("non-arithmetic regular logarithm: log_{%s} (%s)\n", $4->str().c_str(), $6->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(*$4);
		vals.push_back(*$6);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec E_LOG

   |	E_LG expr { // Binary log
    		printf("non-arithmetic binary logarithm of %s\n", $2->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (2));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec E_LG

   |	E_LN expr { // Natural log
    		printf("non-arithmetic natural logarithm of %s\n", $2->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (exp(1.0)));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec E_LN

   |	E_LOG expr { // Log base 10
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (10));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec E_LOG

   |	E_COT expr { // Cot
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cot_op(vals));
} %prec E_CSC

   |	E_SEC expr { // Sec
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sec_op(vals));
} %prec E_CSC

   |	E_CSC expr { // Csc
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::csc_op(vals));
} %prec E_CSC

   |	E_TAN expr { // Tan
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::tan_op(vals));
} %prec TAN

   |	E_COS expr { // Cos
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cos_op(vals));
} %prec E_COS

   |	E_SIN expr { // Sin
		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sin_op(vals));
} %prec E_SIN;

/* Direct Operand: dependant, scalar or parenthesized expression */
dopn: 	dopn E_SUPERSCRIPT dopn {
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals));
} %prec E_SUPERSCRIPT

    |	dpnt {
    		$$ = $1;
}

    |	sclr {
    		$$ = $1;
}

    |	prth {
    		$$ = $1;
};

/* Dependant: variable, function */
dpnt:	E_IDENT { // Variable
    		printf("dependant, variable %s\n", $1);
		variable <double> var;
		string str = $1;
		
		try {
			var = vst.find(str);
		} catch (...) {
			yyerror(value, vst, "no variable in scope");
		}

		$$ = new operand <double> (var.get());
};

/* Scalar: pure numerical values */
sclr:	E_NUMBER { // Number
		$$ = new operand <double> ($1);
    		printf("scalar, %s\n", $$->str().c_str());
};

/* Parenthesis: parenthesized expressions */
prth:	E_LPAREN expr E_RPAREN { // Parenthesis
    		printf("parenthesis, %s\n", $2->str().c_str());
   		$$ = $2;
} %prec E_LPAREN;
   
%%

template <class T>
void yyerror (operand <T> *optr, var_stack <T> vst, const char *error)
{
	cout << error << endl;
}
