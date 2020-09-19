// C/C++ headers
#include <algorithm>
#include <iostream>

// OpenGL headers
#include <GL/freeglut.h>
#include <GL/gl.h>

// Engine headers
#include <vector.hpp>

// Macros
#define vector_vertex_2f(x) glVertex2f(x[0], x[1])

// Namespaces
using namespace std;

// Typedefs
typedef Vector <double> Vec;

// Constants
const double pi = acos(-1);
const double dt = 10;
const double unit = 1E-5/dt;

const double gm = -9.81;

const Vec ga = {0, gm};

// Global variables
double h = 1;
double y = 0;
double l = 0.5;

// Velocity
Vec v = {0, 0};

// Net force
Vec F = {0, 0};

// Mass in kg
double m = 0.1;

// Ellasticity
double e = 0.6;

// Line angle
double theta = pi/3;

// Angular speed
double dtheta = 0;

Vec M = {0.2, 0.5};

auto S = [&]() {return M + Vec::rarg(l/2, theta);};
auto E = [&]() {return M - Vec::rarg(l/2, theta);};

Vec down = {0, 1};

// Draw function
void draw()
{
	// Setup
	glClearColor(0.4, 0.4, 0.4, 0.4);
	glClear(GL_COLOR_BUFFER_BIT);

	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	// Body
	glBegin(GL_LINES);
		glColor3f(0.9, 1.0, 0.9);
		vector_vertex_2f(S());
		vector_vertex_2f(E());
	glEnd();
	
	// Force vector
	Vec Mf = M + (F/(unit)).normalized()/10.0;
	glBegin(GL_LINES);
		glColor3f(0, 1, 0);
		vector_vertex_2f(M);
		vector_vertex_2f(Mf);
	glEnd();
	
	glBegin(GL_LINES);
		vector_vertex_2f(Mf);
		vector_vertex_2f((Mf + Vec::rarg(0.02, F.arg() + pi/2 + pi/3)));
	glEnd();
	
	glBegin(GL_LINES);
		vector_vertex_2f(Mf);
		vector_vertex_2f((Mf + Vec::rarg(0.02, F.arg() + pi/2 + 2 * pi/3)));
	glEnd();
	
	// Velocity vector
	Vec Mv = M + (v/unit).normalized()/10.0;
	glBegin(GL_LINES);
		glColor3f(1, 0, 0);
		vector_vertex_2f(M);
		vector_vertex_2f(Mv);
	glEnd();
	
	glBegin(GL_LINES);
		vector_vertex_2f(Mv);
		vector_vertex_2f((Mv + Vec::rarg(0.02, v.arg() + pi/2 + pi/3)));
	glEnd();
	
	glBegin(GL_LINES);
		vector_vertex_2f(Mv);
		vector_vertex_2f((Mv + Vec::rarg(0.02, v.arg() + pi/2 + 2 * pi/3)));
	glEnd();

	// Ground
	glBegin(GL_LINES);
		glColor3f(0.5, 0.5, 0.5);
		glVertex2f(-1, 0);
		glVertex2f(1, 0);
	glEnd();
	
	// Flush output
	glFlush();
}

// Keyboard input
void keyboard(int key, int x, int y)
{
	if (key == GLUT_KEY_UP) {
		h += 0.01;
	} else if (key == GLUT_KEY_DOWN) {
		h -= 0.01;
	} else if (key == GLUT_KEY_F1) {
		glutDestroyWindow(glutGetWindow());

		exit(0);
	}

	// Request display update
	glutPostRedisplay();
}

double solve(int sgn)
{
	double a;
	double b;

	a = 0;
	b = 1;

	// Scaled y-position
	auto Y = [&](double x) {return 100 * (M[1] + x * v[1] + sgn * l * sin(theta + x * dtheta)/2);};

	double mid;
	while (fabs(Y(a) - Y(b)) > 0.00001) {
		mid = (a + b)/2;

		if (Y(mid) > 0)
			a = mid;
		else
			b = mid;
	}

	/* cout << "d: " << fabs(Y(a) - Y(b)) << endl;
	cout << "\ta: " << a << endl;
	cout << "\t\tY(a): " << Y(a) << endl;
	cout << "\tb: " << b << endl;
	cout << "\t\tY(b): " << Y(b) << endl; */

	return a;
}

// Physics
void update(int value)
{
	// Force
	 F = m * ga;

	if (E()[1] <= 0) {
		F -= Vec::rarg((e * m * gm)/sin(theta), theta);

		dtheta += (2 * e * gm * cos(theta) * dt)/(1000 * pi);
	}
	
	if (S()[1] <= 0) {
		F -= Vec::rarg((e * m * gm)/sin(theta), pi - theta);

		dtheta -= (2 * e * gm * cos(theta) * dt)/(1000 * pi);
	}

	// Apply angular velocity
	theta += dtheta;

	// Scale force
	F *= unit;

	// Add acceleration to the velocity
	v += F/m;

	// Aplpy the velocity
	M += v;

	/* if (E()[1] <= 0) {
		dtheta -= (2 * e * -gm * cos(theta) * dt)/(1000 * l * pi);

		F -= Vec::rarg(e * gm, theta);
	} else if (S()[1] <= 0) {
		dtheta -= (2 * e * gm * cos(theta) * dt)/(1000 * l * pi);

		F -= Vec::rarg(e * gm, -theta);
	}

	if ((S()[1] > 0) && (E()[1] > 0)) {
		// M += k * v;

		F += m * ga;
	}
	
	v += F/m * unit;

	// Time scale
	double k = 1;

	if ((M + Vec::rarg(l/2, theta + dtheta) + v)[1] < 0) {
		cout << "Right Predicted!" << endl;
		k = min(k, solve(1));
	}
	
	if ((M - Vec::rarg(l/2, theta + dtheta) + v)[1] < 0) {
		cout << "Left Predicted!" << endl;
		k = min(k, solve(-1));
	}

	cout << "k: " << k << endl;

	M += v;

	theta += k * dtheta;

	V = Vec::rarg(l/2, theta); */

	glutPostRedisplay();
	glutTimerFunc((int) dt, update, 0);
}

// Main
int main(int argc, char **argv)
{
	// Window setup
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(1000, 1000);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Zhetapi Physics Engine");

	// Functions
	glutSpecialFunc(keyboard);
	glutDisplayFunc(draw);

	// Timers
	glutTimerFunc((int) dt, update, 0);

	// Run main loop
	glutMainLoop();

	return 0;
}
