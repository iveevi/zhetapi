%{
	#include <iostream>
	#include <sstream>
	#include <string>

	using namespace std;
	
	extern "C" int yylex();
%}

DIGIT    [0-9]

%%

{DIGIT}+("/"{DIGIT}+)? {
		yylval.ident = yytext;
		return RATIONAL;
}

{DIGIT}+("/"{DIGIT}+)? {
		yylval.ident = yytext;
		return RATIONAL_COMPLEX;
}

-?([0-9]+|[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?) {
		yylval.ident = yytext;
		return REAL;
}

-?([0-9]+|[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?)i {
		yylval.ident = yytext;
		return REAL_COMPLEX;
}

"log"		{return LOG;}
"ln"		{return LN;}
"lg"		{return LG;}

"sin"		{return SIN;}
"cos"		{return COS;}
"tan"		{return TAN;}
"csc"		{return CSC;}
"sec"		{return SEC;}
"cot"		{return COT;}

"sum"		{return SUM;}

"="		{return EQUALS;}

"^"		{return SUPERSCRIPT;}
"_"		{return SUBSCRIPT;}

"-"		{return MINUS;}
"+"		{return PLUS;}
"*"		{return MULT;}
"/"		{return DIV;}

"!"		{return FACTORIAL;}

"("		{return LPAREN;}
")"		{return RPAREN;}

"{"		{return LBRACE;}
"}"		{return RBRACE;}

"["		{return LBRACKET;}
"]"		{return RBRACKET;}

","		{return SEPARATOR;}

[A-Za-z]+ {
		yylval.ident = yytext;
		return IDENT;
}

[A-Za-z]+_[A-Za-z0-9]+ {
		yylval.ident = yytext;
		return IDENT;
}

"\n"		{return END;}

.		{}

%%
