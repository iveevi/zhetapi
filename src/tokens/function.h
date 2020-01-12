#include "token.h"

namespace tokens {
	/* Beginning of the function class - first complete
	 * toke_tree and ftoken_tree classes
	template <class oper_t>
	class function : public token {
	public:
		typedef oper_t (*ftype)(const std::vector <varible> &);

		function();
		function(std::string, ftype, opers);

		/Use these functions to
		* save space in the function
		* stack class - instead
		* of creating new objects
		* and using space, modify
		* old ones (if the program
		* is sure they wont be used
		* again) *
		void set(std::string);
		void set(function, opers);
		void set(std::string, ftype, opers);

		function get() const;
		function operator~() const;
	private:
		function func;
		std::string name;
		std::size_t opers;
	};

	typedef function <num_t> func_t; */
}
