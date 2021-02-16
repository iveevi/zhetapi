#include "../global.hpp"

// Directions
Vec dirs = acos(-1) * Vec {
	0.5, 1.0, 1.5, 2.0
};

// Heurestic
static double reward(Agent state, double angle, int depth)
{
	if (depth == 0)
		return state.reward();

	// Action
	Vec A = Vec::rarg(state.force, angle);
	
	// Field
	Vec E = F(state.position);

	// Dilate the time
	state.move(A + E, 10 * delta);

	auto ftn = [&](double theta) {
		Agent c = state;

		// Action
		Vec A = Vec::rarg(c.force, theta);
		
		// Field
		Vec E = F(c.position);

		// Dilate the time
		c.move(A + E, 10 * delta);
	
		double r = reward(c, theta, depth - 1);

		return r;
	};

	return state.reward() + max(ftn, dirs);
}

double heurestic(Vec x)
{
	static const int rdepth = 4;			// Recrusion depth
	static auto ftn = [&](double theta) {
		return reward(agent, theta, rdepth);
	};

	return argmax(ftn, dirs);
}
