#ifndef SPECIAL_TOKENS_H_
#define SPECIAL_TOKENS_H_

// Engine headers
#include "../token.hpp"
#include "../operand.hpp"

namespace zhetapi {

#define BLANK_CODE 0x1
#define BREAK_CODE 0x2
#define CONTINUE_CODE 0x3

Token *blank_token();		// Dummy for empty args
Token *break_token();		// Break in loops
Token *continue_token();	// Continue in loops

bool is_blank_token(Token *);
bool is_break_token(Token *);
bool is_continue_token(Token *);

}

#endif