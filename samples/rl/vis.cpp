#include "global.hpp"

// Glut functions
void timer(int value)
{
	step();

	glutPostRedisplay();
	glutTimerFunc(1000 * delta, timer, 1);
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	double r = agent.radius;
	double l = 1.5 * r;

	Vec p = agent.position;
	Vec s = agent.start;

	Vec f = F(p);
	Vec v = agent.velocity;

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
