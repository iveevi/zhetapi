%{
	#include <iostream>
	#include <cctype>
	#include <cstring>
	#include <vector>
	#include <stack>
	#include <string>

	extern "C" int yylex();
  	
	using namespace std;

	void yyerror (string *str, const char *error);
%}

%define parse.error verbose

%parse-param {string *(&str)}

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
	string		*expr;
	string		*coll;
	string		*term;
	string		*felm;
	string		*dopn;
	string		*dpnt;
	string		*prth;
	string		*sclr;

	const char	*ident;
	double       	value;
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
     		str = $1;
		return 0;
};

/* Expression: general exprression */
expr:  	expr SUPERSCRIPT expr { // Exponentiation
   		$$ = new string(*$1 + "^{" + *$3 + "}");
} %prec SUPERSCRIPT

   |	expr MULT expr { // Multiplication
   		$$ = new string(*$1 + " \\cdot " + *$3);
} %prec MULT

   |	expr DIV expr { // Division
   		$$ = new string("\\frac{" + *$1 + "}{" + *$3 + "}");
} %prec DIV

   |	expr PLUS expr { // Addition
   		$$ = new string(*$1 + " + " + *$3);
} %prec PLUS

   |	expr MINUS expr { // Subtraction
   		$$ = new string(*$1 + " - " + *$3);
} %prec MINUS

   | 	MINUS coll {
		$$ = new string("-" + *$2);
} %prec MINUS

   |	coll {
   		$$ = $1;
} %prec LOG;

/* Collective: terms and miscellanics */
coll:	term felm { // Implicit Multiplication: term and non-arithmetic operation
		$$ = new string(*$1 + *$2); 
} %prec LOG

    |	felm {
    		$$ = $1;
} %prec LOG

    |	term {
    		$$ = $1;
} %prec MULT;

/* Term: algebraic term */
term:	term term { // Implicit Multiplication: two or more terms
    		$$ = new string(*$1 + *$2);
} %prec MULT
    		
    |	dopn { // Direct Operand
    		$$ = $1;
};

/* Functional Elementary Operations: non-arithmetic operations */
felm:	LOG SUBSCRIPT LBRACE expr RBRACE expr {
		$$ = new string("\\log_{" + *$4 + "} {" + *$6 + "}");
} %prec LOG

   |	LG expr { // Binary log
		$$ = new string("\\lg{" + *$2 + "}");;
} %prec LG

   |	LN expr { // Natural log
		$$ = new string("\\ln{" + *$2 + "}");;
} %prec LN

   |	LOG expr { // Log base 10
		$$ = new string("\\log{" + *$2 + "}");;
} %prec LOG

   |	COT expr { // Cot
		$$ = new string("\\cot{" + *$2 + "}");;
} %prec CSC

   |	SEC expr { // Sec
		$$ = new string("\\sec{" + *$2 + "}");;
} %prec CSC

   |	CSC expr { // Csc
		$$ = new string("\\csc{" + *$2 + "}");;
} %prec CSC

   |	TAN expr { // Tan
		$$ = new string("\\tan{" + *$2 + "}");;
} %prec TAN

   |	COS expr { // Cos
		$$ = new string("\\cos{" + *$2 + "}");;
} %prec COS

   |	SIN expr { // Sin
		$$ = new string("\\sin{" + *$2 + "}");;
} %prec SIN;

/* Direct Operand: dependant, scalar or parenthesized expression */
dopn: 	dopn SUPERSCRIPT dopn {
	    	$$ = new string(*$1 + "^{" + *$3 + "}");
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
    		string s = string($1);
		if (s == "pi")
			s = "\\pi";
		$$ = new string(s);
};

/* Scalar: pure numerical values */
sclr:	NUMBER { // Number
		$$ = new string(to_string($1));
};

/* Parenthesis: parenthesized expressions */
prth:	LPAREN expr RPAREN { // Parenthesis
   		$$ = new string("\\left(" + *$2 + "\\right)");
} %prec LPAREN;
   
%%

void yyerror (string *str, const char *error)
{
	cout << error << endl;
}
