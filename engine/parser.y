%{
	#include <unordered_map>
	#include <vector>
	#include <string>
	
	extern "C" int yylex();
  	
	void yyerror(stree *(&), const char *);
%}

%define parse.error verbose

%parse-param	{stree *(&root)}

%token IDENT
%token NUMBER

%token PLUS
%token MINUS
%token MULT
%token DIV

%token SIN	COS	TAN
%token CSC	SEC	COT
%token LOG	LN	LG

%token SUPERSCRIPT
%token SUBSCRIPT

%token LPAREN		RPAREN
%token LBRACE 	RBRACE
%token LBRACKET	RBRACKET

%token END

%union {
	stree		*expr;
	stree		*coll;
	stree		*term;
	stree		*felm;
	stree		*dopn;
	stree		*dpnt;
	stree		*prth;
	stree		*sclr;

	const char	*ident;
	const char	*value;
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
%left	PLUS	MINUS
%left	MULT	DIV

%precedence	LBRACKET	RBRACKET
%precedence	LBRACE		RBRACE
%precedence	LPAREN		RPAREN

%precedence	SUPERSCRIPT	SUBSCRIPT
%precedence	SIN	COS	TAN
%precedence	CSC	SEC 	COT
%precedence	LOG	LN	LG

%%

/* make computations based to template type later */

/* Input: general user input */
input:	expr END {
		root = $1;
		return 0;
};

/* Expression: general exprression */
expr:  	expr SUPERSCRIPT expr { // Exponentiation
		$$ = new stree("^", l_operation, {$1, $3});
} %prec SUPERSCRIPT

   |	expr MULT expr { // Multiplication
		$$ = new stree("*", l_operation, {$1, $3});
} %prec MULT

   |	expr DIV expr { // Division
		$$ = new stree("/", l_operation, {$1, $3});
} %prec DIV

   |	expr PLUS expr { // Addition
		$$ = new stree("+", l_operation, {$1, $3});
} %prec PLUS

   |	expr MINUS expr { // Subtraction
		$$ = new stree("-", l_operation, {$1, $3});
} %prec MINUS

   | 	MINUS coll {
		$$ = new stree("*", l_operation, {
			new stree ("-1", l_number, {}), $2
		});
} %prec MINUS

   |	coll {
   		$$ = $1;
} %prec LOG;

/* Collective: terms and miscellanics */
coll:	term felm { // Implicit Multiplication: term and non-arithmetic operation
		$$ = new stree ("*", l_operation, {$1, $2});
} %prec LOG

    |	felm {
    		$$ = $1;
} %prec LOG

    |	term {
    		$$ = $1;
} %prec MULT;

/* Term: algebraic term */
term:	term term { // Implicit Multiplication: two or more terms
		$$ = new stree ("*", l_operation, {$1, $2});
} %prec MULT
    		
    |	dopn { // Direct Operand
    		$$ = $1;
};

/* Functional Elementary Operations: non-arithmetic operations */
felm:	LOG SUBSCRIPT LBRACE expr RBRACE expr {
		$$ = new stree("log", l_operation, {$4, $6});
} %prec LOG

   |	LG expr { // Binary log
		$$ = new stree("log", l_operation, {
			new stree("2", l_number, {}), $2
		});
} %prec LG

   |	LN expr { // Natural log
		$$ = new stree("log", l_operation, {
			new stree("e", l_variable_cluster, {}), $2
		});
} %prec LN

   |	LOG expr { // Log base 10
		$$ = new stree("log", l_operation, {
			new stree("10", l_number, {}), $2
		});
} %prec LOG

   |	COT expr { // Cot
		$$ = new stree("cot", l_operation, {$2});
} %prec CSC

   |	SEC expr { // Sec
		$$ = new stree("sec", l_operation, {$2});
} %prec CSC

   |	CSC expr { // Csc
		$$ = new stree("csc", l_operation, {$2});
} %prec CSC

   |	TAN expr { // Tan
		$$ = new stree("tan", l_operation, {$2});
} %prec TAN

   |	COS expr { // Cos
		$$ = new stree("cos", l_operation, {$2});
} %prec COS

   |	SIN expr { // Sin
		$$ = new stree("sin", l_operation, {$2});
} %prec SIN;

/* Direct Operand: dependant, scalar or parenthesized expression */
dopn: 	dopn SUPERSCRIPT dopn {
		$$ = new stree("^", l_operation, {$1, $3});
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
    		$$ = new stree {$1, l_variable_cluster, {}};
};

/* Scalar: pure numerical values */
sclr:	NUMBER { // Number
		$$ = new stree {$1, l_number, {}};
};

/* Parenthesis: parenthesized expressions */
prth:	LPAREN expr RPAREN { // Parenthesis
   		$$ = $2;
} %prec LPAREN;
   
%%

void yyerror(stree *(&n), const char *error)
{
	std::cout << error << std::endl;
}
