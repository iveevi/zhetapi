#ifndef STRING_TREE_H_
#define STRING_TREE_H_

#include <vector>
#include <string>

/**
 * @brief Enumeration for
 * labeling the stree nodes.
 */
enum st_label {
	l_na,
	l_number,
	l_number_integer,
	l_number_real,
	l_number_rational,
	l_complex_real,
	l_complex_rational,
	l_vector,
	l_matrix,
	l_operation,
	l_variable_cluster,
	l_fn
};

/**
 * @brief Array of corresponding
 * string values of the labels
 * in st_label.
 */
std::string st_str_labs[] = {
	"na",
	"number",
	"number - integer",
	"number - real",
	"number - Rational",
	"complex - real",
	"complex - Rational",
	"vector",
	"matrix",
	"operation",
	"Variable cluster",
	"function"
};

/**
 * @brief A tree data structure
 * used in the intermediatery
 * process of parsing an expression
 * or input in general, which
 * allows templating in the node
 * class to happen without
 * mingling with the flex/bison
 * pair.
 */
class stree {
	std::string rep;

	st_label type;

	std::vector <stree *> leaves;
public:
	stree(const std::string &, st_label,
			const std::vector <stree *> &);

	stree(const std::string &);

	~stree();

	const stree &operator=(const stree &);

	const st_label &kind() const;
	const std::string &str() const;
	const std::vector <stree *> &children() const;
	
	void set(stree *);
	void set_children(const std::vector <stree *> &);

	void print(int = 1, int = 0) const;
};

#include <parser.tab.c>
#include <lex.yy.c>

stree::stree(const std::string &str, st_label lab,
		const std::vector <stree *> &lv)
	: rep(str), type(lab), leaves(lv) {}

stree::stree(const std::string &str)
{
	char *stripped;
	size_t i;
	
	stripped = new char[str.length() + 1];
	for (i = 0; i < str.length(); i++)
		stripped[i] = str[i];
	stripped[i] = '\n';

	yy_scan_string(stripped);

	stree *out;

	yyparse(out);

	*this = *out;

	delete[] stripped;
	delete out;
}

stree::~stree()
{
	for (size_t i = 0; i < leaves.size(); i++)
		delete leaves[i];
}

const stree &stree::operator=(const stree &other)
{
	if (this != &other) {
		rep = other.rep;
		type = other.type;

		leaves.clear();
		for (auto s : other.leaves) {
			stree *tmp = new stree("", l_na, {});
			*tmp = *s;

			leaves.push_back(tmp);
		}
	}

	return *this;
}

const st_label &stree::kind() const
{
	return type;
}

const std::string &stree::str() const
{
	return rep;
}

const std::vector <stree *> &stree::children() const
{
	return leaves;
}

void stree::set(stree *other)
{
	rep = other->rep;
	type = other->type;
	leaves = other->leaves;
}

void stree::set_children(const std::vector <stree *> &other)
{
	leaves = other;
}

void stree::print(int num, int lev) const
{
	int counter = lev;
	while (counter > 0) {
		std::cout << "\t";
		counter--;
	}

	std::cout << "#" << num << " - [" << st_str_labs[type] << "] "
		<< rep << " @ " << this << std::endl;

	counter = 0;
	for (stree *itr : leaves) {
		if (itr == nullptr)
			continue;
		itr->print(++counter, lev + 1);
	}
}

#endif
