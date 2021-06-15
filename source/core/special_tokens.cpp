#include "../../engine/core/special_tokens.hpp"

namespace zhetapi {

Token *blank_token()
{
	static const Operand <Token *> t_blank((Token *) BLANK_CODE);

	return t_blank.copy();
}

Token *break_token()
{
	static const Operand <Token *> t_break((Token *) BREAK_CODE);

	return t_break.copy();
}

Token *continue_token()
{
	static const Operand <Token *> t_continue((Token *) CONTINUE_CODE);

	return t_continue.copy();
}

bool is_blank_token(Token *tptr)
{
	static const Operand <Token *> t_blank((Token *) BLANK_CODE);

	return t_blank == tptr;
}

bool is_break_token(Token *tptr)
{
	static const Operand <Token *> t_break((Token *) BREAK_CODE);

	return t_break == tptr;
}

bool is_continue_token(Token *tptr)
{
	static const Operand <Token *> t_continue((Token *) CONTINUE_CODE);

	return t_continue == tptr;
}

}
