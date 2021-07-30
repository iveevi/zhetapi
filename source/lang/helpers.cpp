#include <core/common.hpp>
#include <stdexcept>

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

	Args out;
	size_t n;

	n = str.length();

	std::string tmp;
	for (size_t i = 0; i < n; i++) {
		if (!quoted) {
			bool ignore = false;
			bool opeq = false;

			// TODO: clean?
			if (pc == '>' || pc == '<' || pc == '!'
				|| (i > 0 && str[i - 1] == '='))
				ignore = true;
			
			if (pc == '+' || pc == '-' || pc == '*' || pc == '@'
				|| pc == '/' || pc == '%'
				|| pc == '&' || pc == '|'
				|| (i > 0 && str[i - 1] == '='))
				opeq = true;

			if (opeq && str[i] == '=') {
				// Grab the rest of the string

				// TODO: error handling
				// if (out.size()) throw
				tmp = tmp.substr(0, tmp.length() - 1);
				return {tmp, tmp + pc + '(' + str.substr(i + 1) + ')'};
			} if (!ignore && str[i] == '=') {
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
			// TODO: start with this
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

Args comma_split(const std::string &str, bool ignore_space = true)
{
	bool quoted = false;

	Args out;

	char c;
	std::string tmp;
	for (size_t i = 0; i < str.length(); i++) {
		c = str[i];
		if (quoted) {
			if (c == '\"')
				quoted = false;

			tmp += c;
		} else {
			if (c == '\"')
				quoted = true;

			if (c == ',' && !tmp.empty()) {
				out.push_back(tmp);

				tmp.clear();
			} else if (!ignore_space || !isspace(c)) {
				tmp += c;
			}
		}
	}

	if (!tmp.empty())
		out.push_back(tmp);

	return out;
}

std::pair <std::string, std::string> as_split(const std::string &str)
{
	std::string libname;
	std::string middle;
	std::string alias;

	size_t i = 0;
	size_t size = str.size();

	while (isspace(str[i]) && i < size) i++;
	while (!isspace(str[i]) && i < size)
		libname += str[i++];
	
	while (isspace(str[i]) && i < size) i++;
	while (!isspace(str[i]) && i < size)
		middle += str[i++];
	
	if (middle == "as") {
		while (isspace(str[i]) && i < size) i++;
		while (!isspace(str[i]) && i < size)
			alias += str[i++];
	} else if (!middle.empty()) {
		throw std::runtime_error("Unexpected \"" + middle + "\" in import clause");
	}

	if (alias.empty())
		alias = libname;

	return {libname, alias};
}

std::pair <std::string, Args> from_split(const std::string &str)
{
	std::string libname;
	std::string middle;
	std::string commas;

	Args all;

	size_t i = 0;
	size_t size = str.size();

	while (isspace(str[i]) && i < size) i++;
	while (!isspace(str[i]) && i < size)
		libname += str[i++];
	
	while (isspace(str[i]) && i < size) i++;
	while (!isspace(str[i]) && i < size)
		middle += str[i++];
	
	if (middle == "import") {
		while (isspace(str[i]) && i < size) i++;
		while (i < size)
			commas += str[i++];
	} else if (!middle.empty()) {
		throw std::runtime_error("Unexpected \"" + middle + "\" in import clause");
	}

	if (commas.empty())
		throw std::runtime_error("Cannot import nothing from library");

	return {libname, comma_split(commas)};
}

}
