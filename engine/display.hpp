#ifndef DISPLAY_H_
#define DISPLAY_H_

#ifndef __AVR	// Does not support AVR

#include <cstdint>

#endif		// Does not support AVR

namespace zhetapi {

/**
 * Display:
 *
 * Display is a struct of display options during neural network training.
 */
struct Display {
	typedef uint8_t type;
	
	static const uint8_t epoch;
	static const uint8_t batch;
	static const uint8_t graph;
};

}

#endif
