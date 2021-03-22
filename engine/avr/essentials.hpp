#ifndef AVR_ESSENTIALS_H_
#define AVR_ESSENTIALS_H_

#ifdef __AVR

#define __avr_switch(code)              

#else

#define __avr_switch(code)      code

#endif

#endif