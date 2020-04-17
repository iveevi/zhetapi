%{
	#include <unordered_map>
	#include <vector>
	#include <string>
	
	#include "func_stack.h"
	#include "var_stack.h"

	#include "variable.h"
	#include "operation.h"
	#include "defaults.h"
  	#include "operand.h"
	// #include "node.h"

	extern "C" int yylex();
  	
	using var = variable <double>;
	using opd = operand <double>;

	using def = defaults <double>;

	using variables = std::unordered_map <std::string, std::vector <node <double> *>>;

	using params = std::vector <variable <double>>;

	// combine var_stack and func_stack into a single stacks/table class
	// and remove the template splay_stack class
	void yyerror(node <double> *(&), params, variables &, var_stack <double> &, func_stack <double> &, const char *);
%}

%define parse.error verbose

%parse-param	{node <double> *(&root)}
%parse-param	{params list}
%parse-param	{variables &vmap}
%parse-param	{var_stack <double> &vst}
%parse-param	{func_stack <double> &fst}

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
	node <double>	*expr;
	node <double>	*coll;
	node <double>	*term;
	node <double>	*felm;
	node <double>	*dopn;
	node <double>	*dpnt;
	node <double>	*prth;
	node <double>	*sclr;

	const char	*ident;
	double    	value;
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
		$$ = new node <double> (&def ::exp_op, {$1, $3});
} %prec SUPERSCRIPT

   |	expr MULT expr { // Multiplication
		$$ = new node <double> (&def ::mult_op, {$1, $3});
} %prec MULT

   |	expr DIV expr { // Division
		$$ = new node <double> (&def ::div_op, {$1, $3});
} %prec DIV

   |	expr PLUS expr { // Addition
		$$ = new node <double> (&def ::add_op, {$1, $3});
} %prec PLUS

   |	expr MINUS expr { // Subtraction
		$$ = new node <double> (&def ::sub_op, {$1, $3});
} %prec MINUS

   | 	MINUS coll {
		$$ = new node <double> (&def ::sub_op, {
			new node <double> (new opd(-1), {}), $2
		});
} %prec MINUS

   |	coll {
   		$$ = $1;
} %prec LOG;

/* Collective: terms and miscellanics */
coll:	term felm { // Implicit Multiplication: term and non-arithmetic operation
    		//printf("collective, term (%s) implicitly multiplicied with non-arithmetic operation (%s)\n", $1->tok->str().c_str(), $2->tok->str().c_str());
		/* std::vector <opd> vals;
		vals.push_back(*$1);
		vals.push_back(*$2);

		$$ = new opd(def ::mult_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($1);
		leaves.push_back($2);

		$$ = new node <double> (&def ::mult_op, l_none, leaves);
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
		std::vector <node <double> *> leaves;
		leaves.push_back($1);
		leaves.push_back($2);

		$$ = new node <double> (&def ::mult_op, l_none, leaves);
} %prec MULT
    		
    |	dopn { // Direct Operand
    		//printf("term with direct operand %s\n", $1->tok->str().c_str());
    		$$ = $1;
};

/* Functional Elementary Operations: non-arithmetic operations */
felm:	LOG SUBSCRIPT LBRACE expr RBRACE expr {
    		// printf("non-arithmetic regular logarithm: log_{%s} (%s)\n", $4->str().c_str(), $6->str().c_str());
   		/* std::vector <opd> vals;
		
		vals.push_back(*$4);
		vals.push_back(*$6);

		$$ = new opd(def ::log_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($4);
		leaves.push_back($6);

		$$ = new node <double> {&def ::log_op,
			l_none, leaves};
} %prec LOG

   |	LG expr { // Binary log
    		//printf("non-arithmetic binary logarithm of %s\n", $2->str().c_str());
   		/* std::vector <opd> vals;
		
		vals.push_back(opd(2));
		vals.push_back(*$2);

		$$ = new opd(def ::log_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back(new node <double> (new opd(2), l_none,
			std::vector <node <double> *> ()));
		leaves.push_back($2);

		$$ = new node <double> {&def ::log_op,
			l_none, leaves};
} %prec LG

   |	LN expr { // Natural log
    		//printf("non-arithmetic natural logarithm of %s\n", $2->tok->str().c_str());
   		/* std::vector <opd> vals;
		
		vals.push_back(opd(exp(1.0)));
		vals.push_back(*$2);

		$$ = new opd(def ::log_op(vals));
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::sin_op,
			l_none, leaves}; */
		std::vector <node <double> *> leaves;
		leaves.push_back(new node <double> (new opd(exp(1)), l_none,
			std::vector <node <double> *> ()));
		leaves.push_back($2);

		$$ = new node <double> {&def ::log_op,
			l_none, leaves};
} %prec LN

   |	LOG expr { // Log base 10
   		/* std::vector <opd> vals;
		
		vals.push_back(opd(10));
		vals.push_back(*$2);

		$$ = new opd(def ::log_op(vals));
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::sin_op,
			l_none, leaves}; */
		std::vector <node <double> *> leaves;
		leaves.push_back(new node <double> (new opd(10), l_none,
			std::vector <node <double> *> ()));
		leaves.push_back($2);

		$$ = new node <double> {&def ::log_op,
			l_none, leaves};
} %prec LOG

   |	COT expr { // Cot
   		/* std::vector <opd> vals;
		vals.push_back(*$2);

		$$ = new opd(def ::cot_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::cot_op,
			l_none, leaves};
} %prec CSC

   |	SEC expr { // Sec
   		/* std::vector <opd> vals;
		vals.push_back(*$2);

		$$ = new opd(def ::sec_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::sec_op,
			l_none, leaves};
} %prec CSC

   |	CSC expr { // Csc
   		/* std::vector <opd> vals;
		vals.push_back(*$2);

		$$ = new opd(def ::csc_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::csc_op,
			l_none, leaves};
} %prec CSC

   |	TAN expr { // Tan
   		/* std::vector <opd> vals;
		vals.push_back(*$2);

		$$ = new opd(def ::tan_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::tan_op,
			l_none, leaves};
} %prec TAN

   |	COS expr { // Cos
   		/* std::vector <opd> vals;
		vals.push_back(*$2);

		$$ = new opd(def ::cos_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::cos_op,
			l_none, leaves};
} %prec COS

   |	SIN expr { // Sin
		/* std::vector <opd> vals;
		vals.push_back(*$2);

		$$ = new opd(def ::sin_op(vals)); */
		std::vector <node <double> *> leaves;
		leaves.push_back($2);

		$$ = new node <double> {&def ::sin_op,
			l_none, leaves};
} %prec SIN;

/* Direct Operand: dependant, scalar or parenthesized expression */
dopn: 	dopn SUPERSCRIPT dopn {
		/* std::vector <opd> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new opd(def ::exp_op(vals)); */
		
		std::vector <node <double> *> leaves;
		leaves.push_back($1);
		leaves.push_back($3);

		$$ = new node <double> {&def ::exp_op,
			l_none, leaves};
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
		std::string str = $1;
		
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
		node <double> *save;
		node <double> *temp;
		node <double> *in;
		// variable <double> var;

		$$ = new node <double> {&def ::mult_op,
			l_none, std::vector <node <double> *>
			{new node <double> {new opd(1), l_none, {}},
			new node <double> {new opd(1), l_none, {}}}};
		temp = $$;

		int num = 0;

		std::string acc;
		for (int i = 0; i < str.length(); i++) {
			acc += str[i];

			auto var = find_if(list.begin(), list.end(),
				[&](const variable <double> &vr) {
					return vr.symbol() == acc;
				}
			);

			if (var != list.end()) {
				/* in = new node <double> {&(*var), l_none, {}};
				temp->leaves[1] = new node <double> {&def ::mult_op,
					l_none, std::vector <node <double> *> {in,
					new node <double> {new opd(1), l_none, {}}}};
				temp = temp->leaves[1]; */
				$$ = new node <double> {&def ::mult_op, l_none,
					{$$, new node <double> {new variable <double> {var->symbol(), true}, l_none, {}}}};
				vmap[var->symbol()].push_back($$->child_at(1));
				// temp = $$;
				acc.clear();
				num++;
			}
		}

		// printf("done\n");
		//$$ = new node <double> {new variable <double> {"x", true}, l_none, {}};
		//print($$, 1, 0);

		if (!num)
			throw node <double> ::invalid_definition();

		// opdval = new opd(var.get());
};

/* Scalar: pure numerical values */
sclr:	NUMBER { // Number
		$$ = new node <double> {new opd($1), {}};
};

/* Parenthesis: parenthesized expressions */
prth:	LPAREN expr RPAREN { // Parenthesis
   		$$ = $2;
} %prec LPAREN;
   
%%

void yyerror(node <double> *(&n), params p, variables &v,
	var_stack &vst, func_stack &fst, const char *error)
{
	std::cout << error << std::endl;
}
