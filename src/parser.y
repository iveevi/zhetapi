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

%define parse.error verbose

%parse-param {operand <double> *value}
%parse-param {var_stack <double> vst}

%token IDENT
%token NUMBER

%token PLUS
%token MINUS
%token MULT
%token DIV

%token SIN COS TAN
%token CSC SEC COT
%token LOG LN LG

%token SUPERSCRIPT
%token SUBSCRIPT

%token LPAREN RPAREN
%token LBRACE RBRACE
%token LBRACKET RBRACKET

%token END

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
%type	<value>	NUMBER
%type	<ident>	IDENT

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
%left PLUS MINUS
%left MULT DIV

%precedence LBRACKET RBRACKET
%precedence LBRACE RBRACE
%precedence LPAREN RPAREN

%precedence SUPERSCRIPT
%precedence SIN COS TAN
%precedence CSC SEC COT
%precedence LOG LN LG

%%

/* make computations based to template type later */

/* Input: general user input */
input:	expr END {
     		value->set($1->get());
		return 0;
};

/* Expression: general exprression */
expr:  	expr SUPERSCRIPT expr { // Exponentiation
   		printf("expression exponentiation\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals));
} %prec SUPERSCRIPT

   |	expr MULT expr { // Multiplication
   		printf("expression multiplication\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec MULT

   |	expr DIV expr { // Division
   		printf("expression divition\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::div_op(vals));
} %prec DIV

   |	expr PLUS expr { // Addition
   		printf("expression addition\n");
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::add_op(vals));
} %prec PLUS

   |	expr MINUS expr { // Subtraction
   		printf("expression substraction\n");
   		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::sub_op(vals));
} %prec MINUS

   | 	MINUS coll {
   		printf("expression negative collective\n");
   		vector <operand <double>> vals;
		vals.push_back(operand <double> (-1));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec MINUS

   |	coll {
   		printf("expression collective\n");
   		$$ = $1;
} %prec LOG;

/* Collective: terms and miscellanics */
coll:	term felm { // Implicit Multiplication: term and non-arithmetic operation
    		printf("collective, term (%s) implicitly multiplicied with non-arithmetic operation (%s)\n", $1->str().c_str(), $2->str().c_str());
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec LOG

    |	felm {
    		$$ = $1;
} %prec LOG

    |	term {
    		printf("collective as a regular term (%s)\n", $1->str().c_str());
    		$$ = $1;
} %prec MULT;

/* Term: algebraic term */
term:	term term { // Implicit Multiplication: two or more terms
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals));
} %prec MULT
    		
    |	dopn { // Direct Operand
    		$$ = $1;
};

/* Functional Elementary Operations: non-arithmetic operations */
felm:	LOG SUBSCRIPT LBRACE expr RBRACE expr {
    		printf("non-arithmetic regular logarithm: log_{%s} (%s)\n", $4->str().c_str(), $6->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(*$4);
		vals.push_back(*$6);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec LOG

   |	LG expr { // Binary log
    		printf("non-arithmetic binary logarithm of %s\n", $2->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (2));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec LG

   |	LN expr { // Natural log
    		printf("non-arithmetic natural logarithm of %s\n", $2->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (exp(1.0)));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec LN

   |	LOG expr { // Log base 10
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (10));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
} %prec LOG

   |	COT expr { // Cot
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cot_op(vals));
} %prec CSC

   |	SEC expr { // Sec
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sec_op(vals));
} %prec CSC

   |	CSC expr { // Csc
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::csc_op(vals));
} %prec CSC

   |	TAN expr { // Tan
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::tan_op(vals));
} %prec TAN

   |	COS expr { // Cos
   		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cos_op(vals));
} %prec COS

   |	SIN expr { // Sin
		vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sin_op(vals));
} %prec SIN;

/* Direct Operand: dependant, scalar or parenthesized expression */
dopn: 	dopn SUPERSCRIPT dopn {
		vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals));
} %prec SUPERSCRIPT

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
dpnt:	IDENT { // Variable
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
sclr:	NUMBER { // Number
		$$ = new operand <double> ($1);
    		printf("scalar, %s\n", $$->str().c_str());
};

/* Parenthesis: parenthesized expressions */
prth:	LPAREN expr RPAREN { // Parenthesis
    		printf("parenthesis, %s\n", $2->str().c_str());
   		$$ = $2;
} %prec LPAREN;
   
%%

template <class T>
void yyerror (operand <T> *optr, var_stack <T> vst, const char *error)
{
	cout << error << endl;
}
