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
		return F_NUMBER;
}

"log"		{return F_LOG;}
"ln"		{return F_LN;}
"lg"		{return F_LG;}

"sin"		{return F_SIN;}
"cos"		{return F_COS;}
"tan"		{return F_TAN;}
"csc"		{return F_CSC;}
"sec"		{return F_SEC;}
"cot"		{return F_COT;}


"^"		{return F_SUPERSCRIPT;}
"_"		{return F_SUBSCRIPT;}

"-"		{return F_MINUS;}
"+"		{return F_PLUS;}
"*"		{return F_MULT;}
"/"		{return F_DIV;}

"("		{return F_LPAREN;}
")"		{return F_RPAREN;}

"{"		{return F_LBRACE;}
"}"		{return F_RBRACE;}

"["		{return F_LBRACKET;}
"]"		{return F_RBRACKET;}

[A-Z,a-z]+ {
		yylval.ident = yytext;
		return F_IDENT;
}

"\n"		{return F_END;}

.		{}

%%
