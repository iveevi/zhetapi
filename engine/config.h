#ifndef CONFIG_H_
#define CONFIG_H_

#include <string>

/**
 * @brief A class used as a method
 * of storing the common set of
 * operations and real/complex space
 * constants.
 */
template <class T>
struct config {
public:
	using opd = operand <T>;
	using opn = operation <operand <T>>;

	opn add;
	opn sub;
	opn mult;
	opn div;
	opn exp;
	opn mod;

	opn sin;
	opn cos;
	opn tan;
	opn csc;
	opn sec;
	opn cot;

	opn log;

	/* Real/complex space
	 * constants */
	T zero;
	T one;
	T neg;

	// Add formatter later
	T read(const std::string &);
};

#endif
