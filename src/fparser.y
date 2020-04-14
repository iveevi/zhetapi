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

	#include "functor.h"
	// #include "func_stack.h"

	#include "var_stack.h"
	#include "variable.h"

	extern "C" int yylex();
  	
	using namespace std;

	void yyerror(functor <double> ::node *(&), functor <double> ::param_list, functor <double> ::map, const char *);
%}

// %define api.prefix {f}
%define parse.error verbose

%parse-param	{functor <double> ::node *(&root)}
%parse-param	{functor <double> ::param_list list}
%parse-param	{functor <double> ::map &vmap}

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
	functor <double> ::node		*expr;
	functor <double> ::node		*coll;
	functor <double> ::node		*term;
	functor <double> ::node		*felm;
	functor <double> ::node		*dopn;
	functor <double> ::node		*dpnt;
	functor <double> ::node		*prth;
	functor <double> ::node		*sclr;

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
     		// value->set($1->get());
		//printf("end of input, root is %s\n", $1->tok->str().c_str());
		root = $1;
		// root = new functor <double> ::node {new operand <double> (10), functor <double> ::m_none, vector <functor <double> ::node *> ()};
		// functor <double> ::print($1, 1, 0);
		// functor <double> ::print(root, 1, 0);
		return 0;
};

/* Expression: general exprression */
expr:  	expr SUPERSCRIPT expr { // Exponentiation
   		//printf("expression exponentiation\n");
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($3);

		$$ = new functor <double> ::node {&defaults <double> ::exp_op, functor <double> ::m_none, leaves};
} %prec SUPERSCRIPT

   |	expr MULT expr { // Multiplication
   		//printf("expression multiplication\n");
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::mult_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($3);

		$$ = new functor <double> ::node {&defaults <double> ::mult_op, functor <double> ::m_none, leaves};
} %prec MULT

   |	expr DIV expr { // Division
   		//printf("expression divition\n");
		/* tor <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::div_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($3);

		$$ = new functor <double> ::node {&defaults <double> ::div_op, functor <double> ::m_none, leaves};
} %prec DIV

   |	expr PLUS expr { // Addition
   		//printf("expression addition\n");
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::add_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($3);

		$$ = new functor <double> ::node {&defaults <double> ::add_op, functor <double> ::m_none, leaves};
} %prec PLUS

   |	expr MINUS expr { // Subtraction
   		//printf("expression substraction\n");
   		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::sub_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($3);

		$$ = new functor <double> ::node {&defaults <double> ::sub_op, functor <double> ::m_none, leaves};
} %prec MINUS

   | 	MINUS coll {
   		//printf("expression negative collective\n");
   		/* vector <operand <double>> vals;
		vals.push_back(operand <double> (-1));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (-1), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::sub_op, functor <double> ::m_none, leaves};
} %prec MINUS

   |	coll {
   		//printf("expression collective\n");
   		$$ = $1;
} %prec LOG;

/* Collective: terms and miscellanics */
coll:	term felm { // Implicit Multiplication: term and non-arithmetic operation
    		//printf("collective, term (%s) implicitly multiplicied with non-arithmetic operation (%s)\n", $1->tok->str().c_str(), $2->tok->str().c_str());
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($2);

		$$ = new functor <double> ::node (&defaults <double> ::mult_op, functor <double> ::m_none, leaves);
} %prec LOG

    |	felm {
    		$$ = $1;
} %prec LOG

    |	term {

    		// printf("collective as a regular term (%s)\n", $1->str().c_str());
    		$$ = $1;
} %prec MULT;

/* Term: algebraic term */
term:	term term { // Implicit Multiplication: two or more terms
    		//printf("term with two terms, %s and %s\n", $1->tok->str().c_str(), $2->tok->str().c_str());
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($2);

		$$ = new functor <double> ::node (&defaults <double> ::mult_op, functor <double> ::m_none, leaves);
} %prec MULT
    		
    |	dopn { // Direct Operand
    		//printf("term with direct operand %s\n", $1->tok->str().c_str());
    		$$ = $1;
};

/* Functional Elementary Operations: non-arithmetic operations */
felm:	LOG SUBSCRIPT LBRACE expr RBRACE expr {
    		// printf("non-arithmetic regular logarithm: log_{%s} (%s)\n", $4->str().c_str(), $6->str().c_str());
   		/* vector <operand <double>> vals;
		
		vals.push_back(*$4);
		vals.push_back(*$6);

		$$ = new operand <double> (defaults <double> ::log_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($4);
		leaves.push_back($6);

		$$ = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
} %prec LOG

   |	LG expr { // Binary log
    		//printf("non-arithmetic binary logarithm of %s\n", $2->str().c_str());
   		/* vector <operand <double>> vals;
		
		vals.push_back(operand <double> (2));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (2), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
} %prec LG

   |	LN expr { // Natural log
    		//printf("non-arithmetic natural logarithm of %s\n", $2->tok->str().c_str());
   		/* vector <operand <double>> vals;
		
		vals.push_back(operand <double> (exp(1.0)));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::sin_op,
			m_none, leaves}; */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (exp(1)), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
} %prec LN

   |	LOG expr { // Log base 10
   		/* vector <operand <double>> vals;
		
		vals.push_back(operand <double> (10));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::sin_op,
			functor <double> ::m_none, leaves}; */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (10), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
} %prec LOG

   |	COT expr { // Cot
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cot_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::cot_op,
			functor <double> ::m_none, leaves};
} %prec CSC

   |	SEC expr { // Sec
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sec_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::sec_op,
			functor <double> ::m_none, leaves};
} %prec CSC

   |	CSC expr { // Csc
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::csc_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::csc_op,
			functor <double> ::m_none, leaves};
} %prec CSC

   |	TAN expr { // Tan
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::tan_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::tan_op,
			functor <double> ::m_none, leaves};
} %prec TAN

   |	COS expr { // Cos
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cos_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::cos_op,
			functor <double> ::m_none, leaves};
} %prec COS

   |	SIN expr { // Sin
		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sin_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::sin_op,
			functor <double> ::m_none, leaves};
} %prec SIN;

/* Direct Operand: dependant, scalar or parenthesized expression */
dopn: 	dopn SUPERSCRIPT dopn {
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals)); */
		
		vector <functor <double> ::node *> leaves;
		leaves.push_back($1);
		leaves.push_back($3);

		$$ = new functor <double> ::node {&defaults <double> ::exp_op,
			functor <double> ::m_none, leaves};
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
    		//printf("dependant, variable %s\n", $1);
		string str = $1;
		
		/* try {
			var = vst.find(str);
		} catch (...) {
			yyerror(value, vst, "no variable in scope");
		} */

		/* bool param = false;
		for (auto v : list) {
			if (v.symbol() == str) {
				param = true;
				break;
			}
		}

		if (!param)
			yyerror(root, list, vmap, "no variable in function scope"); */

		// variable <double> *var = new variable <double> {str, true};
		functor <double> ::node *save;
		functor <double> ::node *temp;
		functor <double> ::node *in;
		// variable <double> var;

		$$ = new functor <double> ::node {&defaults <double> ::mult_op,
			functor <double> ::m_variable, vector <functor <double> ::node *>
			{new functor <double> ::node {new operand <double> (1), functor <double> ::m_none, {}},
			new functor <double> ::node {new operand <double> (1), functor <double> ::m_none, {}}}};
		temp = $$;

		int num = 0;

		string acc;
		for (int i = 0; i < str.length(); i++) {
			acc += str[i];

			auto var = find_if(list.begin(), list.end(),
				[&](const variable <double> &vr) {
					return vr.symbol() == acc;
				}
			);

			if (var != list.end()) {
				/* in = new functor <double> ::node {&(*var), functor <double> ::m_none, {}};
				temp->leaves[1] = new functor <double> ::node {&defaults <double> ::mult_op,
					functor <double> ::m_none, vector <functor <double> ::node *> {in,
					new functor <double> ::node {new operand <double> (1), functor <double> ::m_none, {}}}};
				temp = temp->leaves[1]; */
				$$ = new functor <double> ::node {&defaults <double> ::mult_op, functor <double> ::m_none,
					{$$, new functor <double> ::node {new variable <double> {var->symbol(), true}, functor <double> ::m_none, {}}}};
				vmap[var->symbol()].push_back($$->leaves[1]);
				// temp = $$;
				acc.clear();
				num++;
			}
		}

		// printf("done\n");
		//$$ = new functor <double> ::node {new variable <double> {"x", true}, functor <double> ::m_none, {}};
		//functor <double> ::print($$, 1, 0);

		if (!num)
			throw functor <double> ::invalid_definition();

		// operand <double> val = new operand <double> (var.get());
};

/* Scalar: pure numerical values */
sclr:	NUMBER { // Number
		operand <double> *val = new operand <double> ($1);
		$$ = new functor <double> ::node {val, functor <double> ::m_constant,
			vector <functor <double> ::node *> ()};
    		//printf("scalar, %s\n", $$->tok->str().c_str());
};

/* Parenthesis: parenthesized expressions */
prth:	LPAREN expr RPAREN { // Parenthesis
    		//printf("parenthesis, %s\n", $2->tok->str().c_str());
   		$$ = $2;
} %prec LPAREN;
   
%%

void yyerror(functor <double> ::node *(&nd), functor <double> ::param_list l, functor <double> ::map m, const char *error)
{
	cout << error << endl;
}
