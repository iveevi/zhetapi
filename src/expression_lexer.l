%{
	#include <iostream>
	#include <sstream>
	#include <string>

	using namespace std;
	
	extern "C" int yylex();
%}

%%

-?([0-9]+|[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?) {
		istringstream iss(yytext);
		iss >> yylval.value;
		return E_NUMBER;
}

"log"		{return E_LOG;}
"ln"		{return E_LN;}
"lg"		{return E_LG;}

"sin"		{return E_SIN;}
"cos"		{return E_COS;}
"tan"		{return E_TAN;}
"csc"		{return E_CSC;}
"sec"		{return E_SEC;}
"cot"		{return E_COT;}


"^"		{return E_SUPERSCRIPT;}
"_"		{return E_SUBSCRIPT;}

"-"		{return E_MINUS;}
"+"		{return E_PLUS;}
"*"		{return E_MULT;}
"/"		{return E_DIV;}

"("		{return E_LPAREN;}
")"		{return E_RPAREN;}

"{"		{return E_LBRACE;}
"}"		{return E_RBRACE;}

"["		{return E_LBRACKET;}
"]"		{return E_RBRACKET;}

[A-Z,a-z]+ {
		yylval.ident = yytext;
		return E_IDENT;
}

"\n"		{return E_END;}

.		{}

%%
