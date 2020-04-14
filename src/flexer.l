%{
	#include <iostream>
	#include <sstream>
	#include <string>

	using namespace std;
	
	extern "C" int yylex();
%}

%option prefix="f"

%%

-?([0-9]+|[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?) {
		istringstream iss(yytext);
		iss >> yylval.value;
		return NUMBER;
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


"^"		{return SUPERSCRIPT;}
"_"		{return SUBSCRIPT;}

"-"		{return MINUS;}
"+"		{return PLUS;}
"*"		{return MULT;}
"/"		{return DIV;}

"("		{return LPAREN;}
")"		{return RPAREN;}

"{"		{return LBRACE;}
"}"		{return RBRACE;}

"["		{return LBRACKET;}
"]"		{return RBRACKET;}

[A-Z,a-z]+ {
		yylval.ident = yytext;
		return IDENT;
}

"\n"		{return END;}

.		{}

%%
