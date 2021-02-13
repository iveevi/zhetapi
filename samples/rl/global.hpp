#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>

#include <GL/glut.h>

#include <vector.hpp>
#include <network.hpp>

#include <std/calculus.hpp>
#include <std/activations.hpp>

using namespace std;
using namespace zhetapi;

// Type aliases
using Vec = Vector <double>;

// Agent and environment structure
struct Agent {
	// Movement
	Vec	velocity;	// Velocity (m/s)
	Vec	position;	// Position (m)
	double	mass;		// Mass (kg)

	// Environment
	Vec	start;
	double	radius;

	// Actions
	Vec	applied;
	Vec	net;
	double	force;
	
	// Initialize
	Agent();

	// Methods
	void reset();
	void move(Vec, double);

	bool in_bounds();

	double distance();
	double reward();
	
	// Miscellaneous
	static double runit();
	static double rintv(double, double);
};

extern Agent agent;
extern Vec dirs;

extern const double delta;

// Environment critical
Vec F(const Vec &);
double Fc(const Vec &, double);

double heurestic(Vec);

void step();

// Visualization critical
void timer(int);
void display();
void reshape(int, int);
