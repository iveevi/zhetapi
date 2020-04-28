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
	using opd = operand <T>;
	using opn = operation <operand <T>>;

	static opn add;
	static opn sub;
	static opn mult;
	static opn div;
	static opn exp;
	static opn mod;

	static opn sin;
	static opn cos;
	static opn tan;
	static opn csc;
	static opn sec;
	static opn cot;

	static opn log;

	/* Real/complex space
	 * constants */
	static T zero;
	static T one;
	static T neg;

	// Add formatter later
	static T read(const std::string &);
};

#endif
