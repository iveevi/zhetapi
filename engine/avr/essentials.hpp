#ifndef AVR_ESSENTIALS_H_
#define AVR_ESSENTIALS_H_

#ifdef _AVR

#define __avr_ignore__(code)
#define __avr_switch__(code1, code2)      code1

#define assert(condition)                                               \
        if (!(condition)) {                                               \
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

#define __avr_ignore__(code)              code
#define __avr_switch__(code1, code2)      code2

using psize_t = std::pair <size_t, size_t>;

#endif

#endif
