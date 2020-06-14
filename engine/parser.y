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
%token	NUMBER

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
	const char		*value;
}

/* Types for the terminal symbols */
%type	<value>	NUMBER
%type	<ident>	IDENT

/* Types for non-terminal symbols */
%type	<tree>	expr
%type	<tree>	sclr
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
%precedence	CSC	SEC 	COT
%precedence	LOG	LN	LG

%precedence	FACTORIAL

%precedence	SEPARATOR

%precedence	EQUALS

%%

input:	expr END {
     	root = $1;
	return 0;
};

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

sclr:	NUMBER {
   	$$ = new stree($1, l_number, {});
};

%%

void yyerror(stree *(&n), const char *error)
{
	std::cout << error << std::endl;
}
