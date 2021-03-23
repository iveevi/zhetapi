#ifndef AVR_ESSENTIALS_H_
#define AVR_ESSENTIALS_H_

#ifdef __AVR

#define __avr_ignore(code)
#define __avr_switch(code1, code2)      code1

#define assert(condition)                                               \
        if (!(condition)) {                                               \
                Serial.println("condtion: \"" #condition "\" failed."); \
                                                                        \
                exit(-1);                                               \
        }

// Pair structure
template <class T, class U>
struct __avr_pair {
	T	first;
	U	second;
};

using psize_t = __avr_pair <size_t, size_t>;

#else

#define __avr_ignore(code)              code
#define __avr_switch(code1, code2)      code2

using psize_t = std::pair <size_t, size_t>;

#endif

#endif
