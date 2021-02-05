#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>

#include <vector.hpp>

#include <std/calculus.hpp>

using namespace std;
using namespace zhetapi;

// Type aliases
using Vec = Vector <double>;

// Force field
auto F = [](const Vec &x) {
	return Vec {
		x[1] + 2 * std::cos(2 * x[1]),
		4 * std::sin(5 * x[1]) * std::cos(10 * x[0]) - x[0]
	};
};

// Average field strength
static double Fc(Vec s, double r)
{
	// Magnitude function
	auto Fm = [&](Vec x) {
		return F(x).norm();
	};

	// Vertical strip
	auto Fm_x = [r, s, Fm](double x) {
		double dx = s.x() - x;
		double delta = sqrt(r * r - dx * dx);

		auto Fm_y = [x, Fm](double y) {
			return Fm({x, y});
		};

		if (delta < 1e-10)
			return 0.0;

		return utility::sv_integral(Fm_y, s.y() - delta, s.y() + delta);
	};

	// Integral
	double I = utility::sv_integral(Fm_x, s.x() - r, s.x() + r);

	// Area
	double A = acos(-1) * r * r;

	return I/A;
}

// Agent and environment structure
struct Agent {
	Agent() {
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
			rintv(0, 1),
			rintv(0, 1)
		};

		position = start;

		// Force
		force = Fc(start, radius);

		// Mass
		mass = rintv(5, 10);
	}

	// Movement
	Vec	velocity;	// Velocity (m/s)
	Vec	position;	// Position (m)
	double	mass;		// Mass (kg)

	// Environment
	Vec	start;
	double	radius;

	// Actions
	double	force;

	// Methods
	void move(Vec F, double dt) {
		Vec a = F/mass;

		// Use the average of initial and final
		// velocities in teh time quanta
		position += 0.5 * (velocity + (velocity + a * dt)) * dt;

		velocity += a * dt;
	}

	double distance() {
		return (position - start).norm();
	}

	bool in_bounds() {
		return distance() <= radius;
	}

	double reward() {
		return 1/distance();
	}

	// Miscellaneous
	double runit() {
		return rand() / ((double) RAND_MAX);
	}

	double rintv(double a, double b) {
		return (a + runit() * (b - a));
	}
};

// Agent
Agent agent;

// Step through each iteration
void step(double dt)
{
	agent.move(F(agent.position), dt);

	cout << "Agent @ " << agent.position << endl;
}

// "Frame time"
const double delta(1.0/60.0);
const chrono::milliseconds frame((int) (1000 * delta));

// Main function
int main()
{
	for (size_t i = 0; i < 360; i++) {
		step(delta);

		this_thread::sleep_for(frame);
	}
}