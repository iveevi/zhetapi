#ifndef AVR_RANDOM_H_
#define AVR_RANDOM_H_

namespace zhetapi {

namespace avr {

struct RandomEngine {
	unsigned long long u;
	unsigned long long v;
	unsigned long long w;

	RandomEngine(unsigned long long j) : v(4101842887655102017LL), w(1) {
		u = j ^ v;
		llint();

		v = u;
		llint();

		w = v;
		llint();
	}

	inline unsigned long long llint() {
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17;
		v ^= v << 31;
		v ^= v >> 8;

		w = 4294957665U * (w & 0xffffffff) + (w >> 32);
		
		unsigned long long x = u ^ (u << 21);
		x ^= x >> 35;
		x ^= x << 4;

		return (x + v) ^ w;
	}

        inline unsigned long lint() {
                return (unsigned long) llint();
        }

	long double ldouble() {
		return 5.42101086242752217E-20 * llint();
	}
};

}

}

#endif