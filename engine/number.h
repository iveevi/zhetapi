#ifndef NUMBER_H_
#define NUMBER_H_

enum set_id {
	s_na,
	s_real,
	s_rational,
	s_complex_real,
	s_complex_rational,
	s_vector_real,
	s_vector_rational,
	s_matrix_real,
	s_matrix_rational
};

std::string _sets[] = {
	"none",
	"real",
	"rational",
	"complex real",
	"complex rational",
	"vector real",
	"vector rational",
	"matrix real",
	"matrix rational"
};

struct number {
	set_id kind = s_na;
};

#endif
