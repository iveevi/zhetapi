#ifndef DISPLAY_H_
#define DISPLAY_H_

// C++ standard headers
#include <cstdint>

/**
 * Display:
 *
 * Display is a struct of display options during neural network training.
 */
struct Display {
	static const uint8_t epoch;
	static const uint8_t batch;
	static const uint8_t graph;
};

#endif
