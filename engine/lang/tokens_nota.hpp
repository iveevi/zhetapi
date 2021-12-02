struct DONE : public LexClass <0> {};
struct PLUS : public LexClass <1> {};
struct MINUS : public LexClass <2> {};
struct TIMES : public LexClass <3> {};
struct DIVIDE : public LexClass <4> {};
struct FACTORIAL : public LexClass <5> {};
struct LOGIC_AND : public LexClass <6> {};
struct LOGIC_OR : public LexClass <7> {};
struct LOGIC_EQ : public LexClass <8> {};
struct PLUS_EQ : public LexClass <9> {};
struct MINUS_EQ : public LexClass <10> {};
struct TIMES_EQ : public LexClass <11> {};
struct DIVIDE_EQ : public LexClass <12> {};
struct BIT_AND : public LexClass <13> {};
struct BIT_OR : public LexClass <14> {};
struct BIT_XOR : public LexClass <15> {};
struct BIT_NOT : public LexClass <16> {};
struct EQ : public LexClass <17> {};
struct NEQ : public LexClass <18> {};
struct GT : public LexClass <19> {};
struct LT : public LexClass <20> {};
struct GTE : public LexClass <21> {};
struct LTE : public LexClass <22> {};
struct ALG : public LexClass <23> {};
struct LPAREN : public LexClass <24> {};
struct RPAREN : public LexClass <25> {};
struct LBRACE : public LexClass <26> {};
struct RBRACE : public LexClass <27> {};
struct LBRACKET : public LexClass <28> {};
struct RBRACKET : public LexClass <29> {};
struct NEWLINE : public LexClass <30> {};
struct COMMA : public LexClass <31> {};
struct ASSIGN_EQ : public LexClass <32> {};
struct IDENTIFIER : public LexClass <33> {};
struct PRIMITIVE : public LexClass <34> {};
struct STRING : public LexClass <35> {};
struct OBJECT : public LexClass <36> {};
struct gr_start : public LexClass <37> {};
struct gr_statements : public LexClass <38> {};
struct gr_statement : public LexClass <39> {};
struct gr_assignment : public LexClass <40> {};
struct gr_expression : public LexClass <41> {};
struct gr_simple_expression : public LexClass <42> {};
struct gr_term : public LexClass <43> {};
struct gr_factor : public LexClass <44> {};
struct gr_full_factor : public LexClass <45> {};
struct gr_closed_factor : public LexClass <46> {};
struct gr_operand : public LexClass <47> {};
