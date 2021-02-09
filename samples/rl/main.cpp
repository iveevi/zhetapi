#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>

#include <GL/glut.h>

#include <vector.hpp>
#include <network.hpp>

#include <std/calculus.hpp>

using namespace std;
using namespace zhetapi;

// Type aliases
using Vec = Vector <double>;

// "Frame time"
const double delta(1.0/120.0);
const chrono::milliseconds frame((int) (1000 * delta));

// Force field
auto F = [](const Vec &x) {
	return Vec {
		0.5 * x.y(),
		-2.0 * x.x()
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
		applied = Vec {
			0.0,
			0.0
		};
		
		net = Vec {
			0.0,
			0.0
		};

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
	Vec	applied;
	Vec	net;
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
		return 1/(distance() + 0.1);
	}

	// Miscellaneous
	static double runit() {
		return rand() / ((double) RAND_MAX);
	}

	static double rintv(double a, double b) {
		return (a + runit() * (b - a));
	}
};

// Agent
Agent agent;

// Networks
ml::NeuralNetwork <double> model(6, {
	ml::Layer <double> (4, new ml::ReLU <double> ())
	ml::Layer <double> (8, new ml::Sigmoid <double> ())
	ml::Layer <double> (4, new ml::ReLU <double> ())
});

ml::NeuralNetwork <double> competence(6, {
	ml::Layer <double> (4, new ml::ReLU <double> ())
	ml::Layer <double> (8, new ml::Sigmoid <double> ())
	ml::Layer <double> (4, new ml::ReLU <double> ())
});

// Heurestic
Vec dirs = acos(-1) * Vec {
	0.5, 1.0, 1.5, 2.0
};

double reward(Agent state, double angle, int depth)
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

// Step through each iteration
void step()
{
	// Action
	Vec A = Vec::rarg(agent.force, heurestic(agent.position));
	
	// Field
	Vec E = F(agent.position);

	agent.move(A + E, delta);

	// Set angle
	agent.applied = A;
	agent.net = A + E;
}

// Glut functions
void timer(int value)
{
	step();

	glutPostRedisplay();
	glutTimerFunc(1000 * delta, timer, 1);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Draw the point

	double r = agent.radius;
	double l = 1.5 * r;

	Vec p = agent.position;
	Vec s = agent.start;

	Vec f = F(p);
	Vec v = agent.velocity;
	Vec a = agent.applied;
	Vec n = agent.net;

	glColor3f(0.0, 1.0, 0.0);
	glBegin(GL_LINES);
		glVertex2f(p.x()/l, p.y()/l);
		glVertex2f((p.x() + f.x())/l, (p.y() + f.y())/l);
	glEnd();
	
	glColor3f(0.0, 1.0, 1.0);
	glBegin(GL_LINES);
		glVertex2f(p.x()/l, p.y()/l);
		glVertex2f((p.x() + v.x())/l, (p.y() + v.y())/l);
	glEnd();
	
	glColor3f(1.0, 0.0, 1.0);
	glBegin(GL_LINES);
		glVertex2f(p.x()/l, p.y()/l);
		glVertex2f((p.x() + a.x())/l, (p.y() + a.y())/l);
	glEnd();
	
	glColor3f(1.0, 0.5, 1.0);
	glBegin(GL_LINES);
		glVertex2f(p.x()/l, p.y()/l);
		glVertex2f((p.x() + n.x())/l, (p.y() + n.y())/l);
	glEnd();

	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_POINTS);
		glVertex2f(s.x()/l, s.y()/l);
		glVertex2f(p.x()/l, p.y()/l);
	glEnd();
	
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_POINTS);
		glVertex2f(0, 0);
	glEnd();

	double turn = 2 * acos(-1) / 100;

	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_LINE_LOOP);
		for (double i = 0; i < 100; i++) {
			glVertex2f(
				(s.x() + r * std::cos(turn * i))/l,
				(s.y() + r * std::sin(turn * i))/l
			);
		}
	glEnd();

	glFlush();
	glutSwapBuffers();
}

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
}

// Main function
int main(int argc, char **argv)
{
	// Remove std requirement for sin and cos
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA
			| GLUT_DEPTH
			| GLUT_DOUBLE
			| GLUT_MULTISAMPLE);

	glEnable(GL_MULTISAMPLE);

	glutInitWindowSize(640, 480);
	glutCreateWindow("Reinforcement Learning Simulation");

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutTimerFunc(1000 * delta, timer, 1);

	glutMainLoop();

	return 0;
	for (size_t i = 0; i < 3600; i++) {

		this_thread::sleep_for(frame);
	}
}
