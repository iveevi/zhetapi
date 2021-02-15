#include "global.hpp"

// Directions
Vec dirs = acos(-1) * Vec {
	0.5, 1.0, 1.5, 2.0
};

// Constructor
Agent::Agent(double gma) : gamma(gma)
{
	srand(clock());

	// Start from resting position
	velocity = Vec {
		0.0,
		0.0
	};

	// Random radius
	radius = rintv(5, 10);

	// Starting position
	start = Vec {
		rintv(-1, 1),
		rintv(-1, 1)
	};

	position = start;

	// Force
	force = Fc(start, radius);

	// Mass
	mass = rintv(5, 10);
}

// Methods
Vec Agent::state()
{
	return concat(position, velocity, F(position));
}

void Agent::reset()
{
	velocity = Vec {
		0.0,
		0.0
	};

	position = start;
}

void Agent::move(Vec F, double dt) 
{
	Vec a = F/mass;

	// Use the average of initial and final
	// velocities in the time quanta
	position += 0.5 * (velocity + (velocity + a * dt)) * dt;

	velocity += a * dt;
}

bool Agent::in_bounds()
{
	return distance() <= radius;
}

double Agent::distance()
{
	return (position - start).norm();
}

double Agent::reward()
{
	return 1/(distance() + 0.5);
}

// Miscellaneous
double Agent::runit()
{
	return rand() / ((double) RAND_MAX);
}

double Agent::rintv(double a, double b)
{
	return (a + runit() * (b - a));
}
