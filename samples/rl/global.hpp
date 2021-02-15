#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include <queue>

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
struct Agent {
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
	Agent(double);

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

// Experience structure (s, a, s', r)
struct experience {
	int	index	= 0;
	bool	done	= false;
	double	reward	= 0;
	double	error	= 0;
	Vec	current	= {};
	Vec	next	= {};

	bool operator<(const experience &e) const {
		return error < e.error;
	}

	bool operator>(const experience &e) const {
		return error > e.error;
	}
};

// Priority replay buffer
class replays : public priority_queue <experience> {
	size_t	__size	= 0;
	size_t	__bsize	= 0;
public:
	replays(size_t size, size_t batch_size) : __size(size),
			__bsize(batch_size) {}

	vector <experience> sample() {
		vector <experience> b;

		for (size_t i = 0; i < __bsize; i++) {
			b.push_back(top());

			pop();
		}

		return b;
	}
	
	void add(const experience &e) {
		if (full())
			replace_bottom(e);
		else
			push(e);
	}

        void replace_bottom(const experience &e) {
                auto it_min = min_element(c.begin(), c.end());

                if (it_min->error < e.error) {
                        *it_min = e;

                        make_heap(c.begin(), c.end(), comp);
                }
        }

	bool full() {
		return (size() == __size);
	}
};

// Forward declaration of shared variables
extern Agent agent;
extern Vec dirs;

extern replays prbf;

extern ml::DNN <double> model;
extern ml::DNN <double> confidence;

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
