#ifndef TYPES_H
#define TYPES_H

namespace types {
	enum sets {};

	struct natural {
		unsigned int val;
	};

	struct integer {
		natural val;
		unsigned int sign : 1;
	};

	struct rational {
		integer num;
		integer denom;
	};

	union real {
		rational rat;
		double dec;
	};

	union complex {
		real a;
		real b;
	};

	struct number {
		union {
			natural n;
			integer z;
			rational q;
			real r;
			complex c;
		};

		sets set;
	};
}

#endif