#ifndef AVR_ESSENTIALS_H_
#define AVR_ESSENTIALS_H_

#ifdef __AVR

#define AVR_IGNORE(code)
#define AVR_SWITCH(code1, code2)      code1

#define assert(condition)                                               \
        if (!(condition)) {                                             \
                Serial.println("condtion: \"" #condition "\" failed."); \
                                                                        \
                exit(-1);                                               \
        }

// Pair structure
template <class T, class U>
struct _avr_pair {
	T	first;
	U	second;
};

using psize_t = _avr_pair <size_t, size_t>;

#else

#define AVR_IGNORE(code)              code
#define AVR_SWITCH(code1, code2)      code2

using psize_t = std::pair <size_t, size_t>;

#endif

#endif
