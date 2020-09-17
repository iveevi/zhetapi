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
const double unit = 0.00001;
const double pi = acos(-1);
const double dt = 25;

const double gm = -9.81;

const Vec ga = {0, gm};

// Global variables
double h = 1;
double y = 0;
double l = 0.5;

// Velocity
Vec v = {0, 0};

// Mass in kg
double m = 0.1;

// Ellasticity
double e = 0.6;

// Line angle
double theta = pi/3;

// Angular speed
double dtheta = 0;

Vec V = Vec::rarg(l/2, theta);
Vec M = {0.2, 0.5};

auto S = [&]() {return M + V;};
auto E = [&]() {return M - V;};

Vec down = {0, 1};

// Draw function
void draw()
{
	// Setup
	glClearColor(0.4, 0.4, 0.4, 0.4);
	glClear(GL_COLOR_BUFFER_BIT);

	glColor3f(0.9, 1.0, 0.9);
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	// Body
	glBegin(GL_LINES);
		vector_vertex_2f(S());
		vector_vertex_2f(E());
	glEnd();

	// Ground
	glBegin(GL_LINES);
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

// Physics
void update(int value)
{
	// Upward force
	Vec F = {0, 0};

	if (E()[1] < 0) {
		dtheta -= (2 * e * -gm * cos(theta) * dt)/(1000 * l * pi);

		F -= Vec::rarg(e * gm, theta);
	} else if (S()[1] < 0) {
		dtheta -= (2 * e * gm * cos(theta) * dt)/(1000 * l * pi);

		F -= Vec::rarg(e * gm, -theta);
	}

	if (((M + V + v)[1] < 0) || ((M - V + v)[1] < 0))
		cout << "PREDICTED" << endl;

	if ((S()[1] > 0) && (E()[1] > 0)) {
		M += v;

		F += m * ga;
	}
	
	v += F/m * unit;

	theta += dtheta;

	V = Vec::rarg(l/2, theta);


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
