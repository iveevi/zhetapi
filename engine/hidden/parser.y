%{
	#include <unordered_map>
	#include <vector>
	#include <string>
	
	#ifdef YYDEBUG
		yydebug = 1;
	#endif
	
	extern "C" int yylex();
  	
	void yyerror(stree *(&), const char *);
%}

%define parse.error verbose

%parse-param	{stree *(&root)}

%token	IDENT
%token	REAL
%token	INTEGER
%token	RATIONAL
%token	REAL_COMPLEX
%token	RATIONAL_COMPLEX

%token	PLUS
%token	MINUS

%token	MULT
%token	DIV

%token	SUM

%token	SIN	COS	TAN
%token	CSC	SEC	COT
%token	LOG	LN	LG

%token	SUPERSCRIPT
%token	SUBSCRIPT

%token	FACTORIAL
%token	DOT

%token	BINOM
%token	CHOOSE

%token	SHUR
%token	TRANSPOSE

%token	LPAREN		RPAREN
%token	LBRACE		RBRACE
%token	LBRACKET	RBRACKET

%token	END
%token	EQUALS
%token	SEPARATOR

%union {
	stree			*tree;
	
	std::vector <stree *>	*pack;

	const char		*ident;
}

/* Types for the terminal symbols */
%type	<ident>	REAL
%type	<ident>	INTEGER
%type	<ident>	RATIONAL
%type	<ident>	REAL_COMPLEX
%type	<ident>	RATIONAL_COMPLEX
%type	<ident>	IDENT

/* Types for non-terminal symbols */
%type	<tree>	expr
%type	<tree>	sclr
%type	<tree>	numr
%type	<tree>	matr
%type	<tree>	vect
%type	<tree>	dpnt
%type	<tree>	term
%type	<tree>	idnt
%type	<tree>	slcl
%type	<tree>	vrcl
%type	<tree>	prth
%type	<tree>	felm
%type	<tree>	func
%type	<tree>	summ
%type	<tree>	coll
%type	<tree>	spex
%type	<tree>	sbex

%type	<pack>	vpck
%type	<pack>	epck
%type	<pack>	pack

/* Precedence information to resolve ambiguity */
%left	PLUS	MINUS
%left	MULT	DIV

%precedence	LBRACKET	RBRACKET
%precedence	LBRACE		RBRACE
%precedence	LPAREN		RPAREN

%precedence	SUM

%precedence	SUPERSCRIPT	SUBSCRIPT

%precedence	SIN	COS	TAN
%precedence	CSC	SEC	COT

%precedence	SINH	COSH	TANH
%precedence	CSCH	SECH 	COTH

%precedence	LOG	LN	LG

%precedence	FACTORIAL

%precedence	SEPARATOR

%precedence	EQUALS

%%

input:	expr END {
     	root = $1;
	return 0;
};

expr:  	expr DOT expr { // Dot Product
	$$ = new stree(".", l_operation, {$1, $3});
} %prec DOT

   |	expr TRANSPOSE { // Exponentiation
	$$ = new stree("transpose", l_operation, {$1});
} %prec TRANSPOSE

   |	expr SHUR expr { // Exponentiation
	$$ = new stree("shur", l_operation, {$1, $3});
} %prec SHUR

   |	expr CHOOSE expr { // Exponentiation
	$$ = new stree("binom", l_operation, {$1, $3});
} %prec CHOOSE

   |	expr BINOM expr { // Exponentiation
	$$ = new stree("binom", l_operation, {$1, $3});
} %prec BINOM

   |	expr SUPERSCRIPT expr { // Exponentiation
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

   |	MINUS coll {
	$$ = new stree("*", l_operation, {
		new stree ("-1", l_number, {}), $2
	});
}

   |	coll {
   	$$ = $1;
};

coll:	term summ {
    	$$ = new stree("*", l_operation, {$1, $2});
}

    |	summ {
    	$$ = $1;
}

    |	term {
    	$$ = $1;
};

term:	sclr dpnt {
    	$$ = new stree("*", l_operation, {$1, $2});
}

    |	dpnt {
    	$$ = $1;
}

    |	slcl {
    	$$ = $1;
};

dpnt:	dpnt SUPERSCRIPT LPAREN expr RPAREN {
    	$$ = new stree("^", l_operation, {$1, $4});
}

    |	dpnt SUPERSCRIPT term {
    	$$ = new stree("^", l_operation, {$1, $3});
}
    
    |	dpnt dpnt {
    	$$ = new stree("*", l_operation, {$1, $2});
}

    |	prth {
    	$$ = $1;
}

    |	felm {
    	$$ = $1;
}

    |	vrcl {
    	$$ = $1;
};

summ:	SUM spex SUBSCRIPT LBRACE dpnt EQUALS expr RBRACE expr {
		$$ = new stree("sum", l_operation, {
			$5,
			$7,
			$2,
			$9
		});
}
    |	SUM SUBSCRIPT LBRACE dpnt EQUALS expr RBRACE spex expr {
		$$ = new stree("sum", l_operation, {
			$4,
			$6,
			$8,
			$9
		});
};

spex:	SUPERSCRIPT LBRACE expr RBRACE {
    		$$ = $3;
};

sbex:	SUBSCRIPT LBRACE expr RBRACE {
    		$$ = $3;
};

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
   
   |	COTH expr { // Cot
		$$ = new stree("coth", l_operation, {$2});
} %prec COTH

   |	SECH expr { // Sec
		$$ = new stree("sech", l_operation, {$2});
} %prec SECH

   |	CSCH expr { // Csc
		$$ = new stree("csch", l_operation, {$2});
} %prec CSCH

   |	TANH expr { // Tan
		$$ = new stree("tanh", l_operation, {$2});
} %prec TANH

   |	COSH expr { // Cos
		$$ = new stree("cosh", l_operation, {$2});
} %prec COSH

   |	SINH expr { // Sin
		$$ = new stree("sinh", l_operation, {$2});
} %prec SINH

   |	felm FACTORIAL {
   	$$ = new stree("!", l_operation, {$1});
};

   |	COT expr { // Cot
		$$ = new stree("cot", l_operation, {$2});
} %prec COT

   |	SEC expr { // Sec
		$$ = new stree("sec", l_operation, {$2});
} %prec SEC

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
} %prec SIN

   |	felm FACTORIAL {
   	$$ = new stree("!", l_operation, {$1});
};

vrcl:	func FACTORIAL {
    	$$ = new stree("!", l_operation, {$1});
}
    |	func {
    	$$ = $1;
}

    |	idnt FACTORIAL {
    	$$ = new stree("!", l_operation, {$1});
}

    |	idnt {
    	$$ = $1;
};

func:	idnt LPAREN pack RPAREN {
    		$$ = $1;

		$$->set_children(*($3));
};

pack:	%empty

    |	expr {
    		$$ = new std::vector <stree *> {$1};
}

    |	pack SEPARATOR expr {
    		$$ = $1;

		$$->push_back($3);
};

prth:	prth FACTORIAL {
    	$$ = new stree("!", l_operation, {$1});
}

    |	LPAREN expr RPAREN {
    	$$ = $2;
};

idnt:	IDENT {
   	$$ = new stree($1, l_variable_cluster, {});
};

slcl:	sclr FACTORIAL {
   	$$ = new stree("!", l_operation, {$1});
}

    |	sclr {
    	$$ = $1;
};
    
sclr:	numr {
    	$$ = $1;
}
    |	matr {
	$$ = $1;
}
    |	vect {
    	$$ = $1;
};

numr:	REAL {
   	$$ = new stree($1, l_number_real, {});
}
    |	INTEGER {
	$$ = new stree($1, l_number_integer, {});
}
    |	RATIONAL {
	$$ = new stree($1, l_number_rational, {});
}
    |	REAL_COMPLEX {
	$$ = new stree($1, l_complex_real, {});
}
    |	RATIONAL_COMPLEX {
	$$ = new stree($1, l_complex_rational, {});
};

matr:	LBRACKET vpck RBRACKET {
    	$$ = new stree("M", l_matrix, *($2));
};

vpck:	vect {
    		$$ = new std::vector <stree *> {$1};
}
    |	vpck SEPARATOR vect {
    		$$ = $1;

		$$->push_back($3);
};

vect:	LBRACKET epck RBRACKET {
    	$$ = new stree("V", l_vector, *($2));
};

epck:	numr {
    		$$ = new std::vector <stree *> {$1};
}
    |	epck SEPARATOR numr {
    		$$ = $1;

		$$->push_back($3);
};

%%

void yyerror(stree *(&n), const char *error)
{
	std::cout << error << std::endl;
}
