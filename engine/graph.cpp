#include <GL/glut.h>

#include "functor.h"

functor <double> f("f(x) = sin x");

/* Sample func itself */
float func(float x)
{
	return x*x;
}

/* Function plotting func */
void draw(float x1, float x2, float y1, float y2, int N)
{
	float x, dx = 1.0/N;

	glPushMatrix(); /* GL_MODELVIEW is default */

	glScalef(1.0 / (x2 - x1), 1.0 / (y2 - y1), 1.0);
	glTranslatef(-x1, -y1, 0.0);
	glColor3f(1.0, 1.0, 1.0);

	glBegin(GL_LINE_STRIP);

	for(x = x1; x < x2; x += dx)
	{
		glVertex2f(x, f(x));
	}

	glEnd();

	glPopMatrix();
};

/* Redrawing func */
void redraw(void)
{

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	draw(-3, 3, 0, 10, 10);

	glutSwapBuffers();
};

/* Idle proc. Redisplays, if called. */
void idle(void)
{
	glutPostRedisplay();
};

/* Key press processing */
void key(unsigned char c, int x, int y)
{
	if(c == 27) exit(0);
};

/* Window reashape */
void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
};

/* Main function */
int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow("Graph plotter");

	/* Register GLUT callbacks. */
	glutDisplayFunc(redraw);
	glutKeyboardFunc(key);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	/* Init the GL state */
	glLineWidth(1.0);
  
	/* Main loop */
	glutMainLoop();
	return 0;
}
