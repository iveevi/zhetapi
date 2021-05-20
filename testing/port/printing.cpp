#include "port.hpp"

// Terminal objects
term_colors reset {"0"};

term_colors bred {"1;31"};
term_colors bgreen {"1;32"};
term_colors byellow {"1;33"};

term_ok ok;
term_err err;

ostream &operator<<(ostream &os, const term_colors &tc)
{
	return (os << "\033[" << tc.color << "m");
}

ostream &operator<<(ostream &os, const term_ok &tok)
{
	return (os << bgreen << "[OK]" << reset);
}

ostream &operator<<(ostream &os, const term_err &terr)
{
	return (os << bred << "[ERR]" << reset);
}