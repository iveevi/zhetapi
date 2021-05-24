#include <core/common.hpp>

namespace zhetapi {

bool is_vaild_ident_char(char c, size_t i)
{
	return (isdigit(c) && i != 0)
		|| isalpha(c)
		|| (c == '$')
		|| (c == '_');
}

bool is_valid_ident(const std::string &ident)
{
	for (size_t i = 0; i < ident.length(); i++) {
		if (!is_vaild_ident_char(ident[i], i))
			return false;
	}

	return true;
}

bool in_args(const Args &target, const Args &suspect)
{
	for (const std::string &arg1 : target) {
		bool safe = false;
		for (const std::string &arg2 : suspect) {
			if (arg1 == arg2) {
				safe = true;

				break;
			}
		}

		if (!safe)
			return false;
	}

	return true;
}

// Extract any arguments from potential idenitifers (of veq)
Args get_args(const std::string &str)
{
	// First argument is the identifier
	std::string ident;

	size_t i = 0;
	for (; i < str.length() && str[i] != '('; i++)
		ident += str[i];

	Args out {ident};
	while (i < str.length()) {
		std::string tmp;

		size_t j = i + 1;
		for (; j < str.length()
				&& str[j] != ','
				&& str[j] != ')'; j++)
			tmp += str[j];

		out.push_back(tmp);
		if (str[j] == ')')
			break;

		if (str[j] != ')' && str[j] != ',')
			return {};

		i = j;
	}

	return out;
}

// TODO: clean
Args eq_split(const std::string &str)
{
	// TODO: dont forget about op=
	// TODO: if possible, split in the first pass
	bool quoted = false;

	char pc = 0;

	std::vector <std::string> out;
	size_t n;

	n = str.length();

	std::string tmp;
	for (size_t i = 0; i < n; i++) {
		if (!quoted) {
			bool ignore = false;

			if (pc == '>' || pc == '<' || pc == '!'
				|| (i > 0 && str[i - 1] == '='))
				ignore = true;
			
			if (!ignore && str[i] == '=') {
				if (i < n - 1 && str[i + 1] == '=') {
					tmp += "==";
				} else if (!tmp.empty()) {
					out.push_back(tmp);

					tmp.clear();
				}
			} else {
				if (str[i] == '\"')
					quoted = true;
				
				tmp += str[i];
			}
		} else {
			if (str[i] == '\"')
				quoted = false;
			
			tmp += str[i];
		}

		pc = str[i];
	}

	if (!tmp.empty())
		out.push_back(tmp);

	return out;
}

}
