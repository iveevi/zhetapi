#include "global.hpp"

// Constructor
Environment::Environment(double gma) : gamma(gma)
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
Vec Environment::state()
{
	return concat(position, velocity, F(position));
}

void Environment::reset()
{
	velocity = Vec {
		0.0,
		0.0
	};

	position = start;
}

void Environment::move(Vec F, double dt) 
{
	Vec a = F/mass;

	// Use the average of initial and final
	// velocities in the time quanta
	position += 0.5 * (velocity + (velocity + a * dt)) * dt;

	velocity += a * dt;
}

bool Environment::in_bounds()
{
	return distance() <= radius;
}

double Environment::distance()
{
	return (position - start).norm();
}

double Environment::reward()
{
	return 1/(distance() + 0.5);
}

// Miscellaneous
double Environment::runit()
{
	return rand() / ((double) RAND_MAX);
}

double Environment::rintv(double a, double b)
{
	return (a + runit() * (b - a));
}
