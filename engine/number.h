#ifndef NUMBER_H_
#define NUMBER_H_

//////////////////////////////////////////
// Forward Declarations
//////////////////////////////////////////

template <class T>
class rational;

template <class T>
class zcomplex;

//////////////////////////////////////////
// Type Checks
//////////////////////////////////////////

template <class>
struct is_rational : public std::false_type {};

template <class T>
struct is_rational <rational <T>> : public std::true_type {};

template <class>
struct is_zcomplex_real : public std::false_type {};

template <class T>
struct is_zcomplex_real <zcomplex <T>> : public std::true_type {};

template <class>
struct is_zcomplex_rational : public std::false_type {};

template <class T>
struct is_zcomplex_rational <zcomplex <rational <T>>> : public std::true_type {};

//////////////////////////////////////////
// Set Labels
//////////////////////////////////////////

enum set_id {
	s_na,
	s_integer,
	s_real,
	s_rational,
	s_complex_real,
	s_complex_rational,
	s_vector_real,
	s_vector_rational,
	s_vector_complex_real,
	s_vector_complex_rational,
	s_matrix_real,
	s_matrix_rational,
	s_matrix_complex_real,
	s_matrix_complex_rational
};

std::string _sets[] = {
	"none",
	"integer",
	"real",
	"rational",
	"complex real",
	"complex rational",
	"vector real",
	"vector rational",
	"vector complex real",
	"vector complex rational",
	"matrix real",
	"matrix rational",
	"matrix complex real",
	"matrix complex rational"
};

/**
 * @brief Base class of all numeric
 * types of computation.
 */
struct number {
	set_id kind = s_na;
};

#endif
