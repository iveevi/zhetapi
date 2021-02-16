#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include <queue>

#include <dlfcn.h>

#include <GL/glut.h>

#include <dnn.hpp>
#include <vector.hpp>

#include <std/calculus.hpp>
#include <std/activations.hpp>
#include <std/optimizers.hpp>
#include <std/erfs.hpp>

using namespace std;
using namespace zhetapi;

// Type aliases
using Vec = Vector <double>;

// Agent and environment structure
struct Environment {
	// Movement
	Vec	velocity;	// Velocity (m/s)
	Vec	position;	// Position (m)
	double	mass;		// Mass (kg)

	// Environment
	Vec	start;
	double	radius;

	// Actions
	double	force;

	// Hyper-parameters
	double	gamma;		// Discount factor
	
	// Initialize
	Environment(double);

	// Methods
	Vec state();

	void reset();
	void move(Vec, double);

	bool in_bounds();

	double distance();
	double reward();
	
	// Miscellaneous
	static double runit();
	static double rintv(double, double);
};

// Strategy representation
struct strategy {
	typedef void (*init_t)(Environment);
	typedef double (*action_t)(Vec);
	typedef void (*reward_t)(double, Vec, bool, double &);

	init_t		h_init		= nullptr;
	action_t	h_action	= nullptr;
	reward_t	h_reward	= nullptr;

	double		reward		= 0;
	double		error		= 0;
	size_t		frames		= 0;
	size_t		tframes		= 0;

	// Ensure that none of the functions are null
	bool validate() {
		return h_init && h_action && h_reward;
	}
};

// Forward declaration of shared variables
extern Environment env;

extern "C" Vec dirs;

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
