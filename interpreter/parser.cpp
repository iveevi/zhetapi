#include "global.hpp"

#define __lineup(c) 		\
	if (c == '\n')		\
		line++;

#define __skip_space()		\
	while (isspace(c = getchar()));

#define __skip_to_char(s)	\
	while ((c = getchar()) != s);

inline bool __file_exists(string file)
{
	if (FILE *f = fopen(file.c_str(), "r")) {
		fclose(f);

		return true;
	} else {
		return false;
	} 
}

// Global scope code
vector <string> global;

// Constants
Operand <bool> *op_true = new Operand <bool> (true);
Operand <bool> *op_false = new Operand <bool> (false);

size_t line = 1;
string file = "";

static int split_for_statement(string condition, string &variable, string &expression)
{
	size_t i = 0;
	size_t i_in = -1;

	size_t len = condition.size();

	// Find the position of 'in'
	while (i < len - 1) {
		if (condition[i] == 'i'
			&& condition[i + 1] == 'n') {
			i_in = i;

			break;
		}

		i++;
	}

	for (size_t i = 0; i < i_in; i++) {
		if (isspace(condition[i]))
			break;
		
		variable += condition[i];
	}

	expression = condition.substr(i_in + 2);

	if (expression.empty())
		return 1;

	return (i_in == -1);
}

static int parse_parenthesized(string &parenthesized)
{
	char c;

	while (isspace(c = getchar())) {
		__lineup(c);
	}

	if (c != '(')
		return -1;

	while ((c = getchar()) != ')')
		parenthesized += c;

	return 0;
}

static int extract_block(string &block)
{
	char c;
	
	__skip_space();
	// while (isspace(c = getchar()));

	if (c == '{') {
		while ((c = getchar()) != '}')
			block += c;
	} else {
		fseek(stdin, -1, SEEK_CUR);

		while ((c = getchar()) != '\n')
			block += c;

		__lineup(c);
	}

	return 0;
}

static int parse_block()
{
	char c;
	
	while (isspace(c = getchar())) {
		__lineup(c);
	}

	if (c == '{') {
		parse('}');
	} else {
		fseek(stdin, -1, SEEK_CUR);

		parse('\n');

		__lineup(c);
	}

	return 0;
}

static int parse_block(string &str)
{
	char c;
	
	while (isspace(c = getchar())) {
		__lineup(c);
	}

	if (c == '{') {
		while ((c = getchar()) != '}') {
			str += c;

			__lineup(c);
		}
	} else {
		fseek(stdin, -1, SEEK_CUR);

		while ((c = getchar()) != '\n')
			str += c;

		__lineup(c);
	}

	return 0;
}

static int parse_block_ignore()
{
	char c;
	
	while (isspace(c = getchar())) {
		__lineup(c);
	}

	if (c == '{') {
		while ((c = getchar()) != '}');
	} else {
		fseek(stdin, -1, SEEK_CUR);

		while ((c = getchar()) != '\n' && c != EOF);

		__lineup(c);
	}

	return 0;
}

static int parse_function(string &ident, vector <string> &params)
{
	char c;

	__skip_space();
	
	fseek(stdin, -1, SEEK_CUR);

	while ((c = getchar()) != '(')
		ident += c;

	string tmp;
	while ((c = getchar()) != ')') {
		if (c == ',') {
			if (!tmp.empty()) {
				params.push_back(tmp);

				tmp.clear();
			}
		} else if (!isspace(c)) {
			tmp += c;
		}
	}

	if (!tmp.empty())
		params.push_back(tmp);

	return 0;
}

bool if_prev = false;
bool if_true = false;

void check(string &keyword)
{
	string parenthesized;
	string block;
	string lname;

	if (keyword == "if") {
		if (parse_parenthesized(parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after an if\n", line);
			exit(-1);
		}

		if_prev = true;

		Token *t = execute(parenthesized);

		if (*t == op_true) {
			if_true = true;

			engine = push_and_ret_stack(engine);

			parse_block();

			engine = pop_and_del_stack(engine);
		} else {
			if_true = false;

			parse_block_ignore();
		}
		
		keyword.clear();
	}

	if (keyword == "elif") {
		if (!if_prev) {
			printf("Error at line %lu: need an if before an elif\n", line);

			exit(-1);
		}

		if (parse_parenthesized(parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after an if\n", line);
			exit(-1);
		}

		Token *t = execute(parenthesized);

		if (*t == op_true) {
			if_true = true;

			engine = push_and_ret_stack(engine);

			parse_block();

			engine = pop_and_del_stack(engine);
		} else {
			if_true = false;

			parse_block_ignore();
		}
		
		keyword.clear();
	}

	if (keyword == "else") {
		if (!if_prev) {
			printf("Error at line %lu: need an if before an else\n", line);

			exit(-1);
		}
		
		if_prev  = false;

		if (if_true)
			parse_block_ignore();
		else
			parse_block();
		
		keyword.clear();
	}

	if (keyword == "while") {
		if (parse_parenthesized(parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after a while\n", line);
			exit(-1);
		}
		
		if (extract_block(block)) {
			printf("Syntax error at line %lu: missing statement after a while\n", line);
			exit(-1);
		}

		engine = push_and_ret_stack(engine);

		Token *t = execute(parenthesized);
		while (*t == op_true) {
			parse(block);
			
			t = execute(parenthesized);
		}
		
		engine = pop_and_del_stack(engine);
		
		keyword.clear();
	}

	if (keyword == "for") {
		if (parse_parenthesized(parenthesized)) {
			printf("Syntax error at line %lu: missing parenthesis after a for\n", line);
			exit(-1);
		}

		string var;
		string expr;
		
		if (split_for_statement(parenthesized, var, expr)) {
			printf("Syntax error at line %lu: unexpected condition in for loop\n", line);
			exit(-1);
		}
		
		if (extract_block(block)) {
			printf("Syntax error at line %lu: missing statement after a for\n", line);
			exit(-1);
		}

		node_manager nm(engine, expr);

		Token *tptr = nm.value(engine);

		// For the love of God make this cleaner
		Operand <std::vector <Token *>> *op = dynamic_cast <Operand <std::vector <Token *>> *> (tptr);

		std::vector <Token *> tok_list = op->get();

		// Push in a new scope
		engine = push_and_ret_stack(engine);

		for (Token *t : tok_list) {
			engine->put(var, t);

			parse(block);
		}

		// Pop scope
		engine = pop_and_del_stack(engine);

		keyword.clear();
	}

	if (keyword == "include") {
		cin >> lname;

		string library = lname + ".zhplib";

		// Check current dir
		string current_dir = "./" + library;
		// string script_dir = __get_dir(file) + library;

		string libfile;

		bool found = false;
		for (string dir : idirs) {
			libfile = dir + "/" + library;

			if (__file_exists(libfile)) {
				import_library(libfile);

				found = true;
				break;
			}
		}

		if (!found) {
			printf("Error at line %lu: could not import library '%s'\n", line, library.c_str());

			exit(-1);
		}

		keyword.clear();
	}

	if (keyword == "alg") {
		// cout << "ALG begin@ " << line << endl;
		string ident;

		vector <string> params;

		if (parse_function(ident, params)) {
			cout << "Failed to parse function..." << endl;

			exit(-1);
		} else {
			parse_block(block);

			// Add a newline to flush parsing
			block += '\n';

			// Create and compile the algorithm structure
			algorithm alg(ident, block, params);
			alg.compile(engine);

			engine->put(alg);
		}

		keyword.clear();

		// cout << "ALG end@ " << line << endl;
	}
}

// Parsing machine
int parse(char ex)
{
	static bool quoted = false;
	static int paren = 0;

	string tmp;
	char c;

	while ((c = getchar()) != ex) {
		if (!quoted) {
			if (c == '\"')
				quoted = true;
			if (c == '(' || c == '{')
				paren++;
			if (c == ')' || c == '}')
				paren--;
			
			if (c == '\n' || (!paren && c == ',')) {
				if (!tmp.empty()) {
					execute(tmp);

					tmp.clear();
				}
			} else if (!isspace(c)) {
				tmp += c;
			}
		} else {
			if (c == '\"')
				quoted = false;
			
			tmp += c;
		}

		__lineup(c);

		// cout << "tmp = " << tmp << endl;
		check(tmp);
	}

	// Flush last instruction
	if (!tmp.empty()) {
		execute(tmp);

		tmp.clear();
	}

	return 0;
}

// Parsing machine
int parse(string str)
{
	static bool quoted = false;
	static int paren = 0;

	string tmp;

	for (size_t i = 0; i < str.length(); i++) {
		char c = str[i];

		if (!quoted) {
			if (c == '\"')
				quoted = true;
			if (c == '(' || c == '{')
				paren++;
			if ((c == ')' || c == '}') && paren)
				paren--;
			
			if (c == '\n' || (!paren && c == ',')) {
				if (!tmp.empty()) {
					// cout << "executing tmp = " << tmp << endl;
					execute(tmp);

					tmp.clear();
				}
			} else if (!isspace(c)) {
				tmp += c;
			}
		} else {
			if (c == '\"')
				quoted = false;
			
			tmp += c;
		}

		__lineup(c);

		// cout << "tmp = " << tmp << endl;
		check(tmp);
	}

	// Flush last instruction
	if (!tmp.empty()) {
		execute(tmp);

		tmp.clear();
	}

	return 0;
}

// Splitting equalities
vector <string> split(string str)
{
	bool quoted = false;

	char pc = 0;

	vector <string> out;
	size_t n;

	n = str.length();

	string tmp;
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
	
	/* cout << "split:" << endl;
	for (auto s : out)
		cout << "\ts = " << s << endl; */

	return out;
}
