#include <iostream>
#include <mutex>
#include <vector>

#include <fixed_vector.hpp>
#include <image.hpp>

// Namespaces
using namespace std;
using namespace zhetapi;

// Forward declarations
class Simulation;
class Shape;
class Rect;

// Add object as a generalization of shape

// Ray class
class Ray {
	// Delta is normalized
	Vec3f	_start;
	Vec3f	_delta;
public:
	Ray(const Vec3f &, const Vec3f &);

	Vec3f at(float);
};

// Camera class
class Camera {
	Vec3f	_position;

	size_t	_width;
	size_t	_height;
public:
	Camera(const Vec3f &, size_t, size_t);

	Ray ray(size_t, size_t) const;
};

// Simulation class
class Simulation {
	Vec3f			_camera;

	image::Image		_screen;

	std::string		_name;

	sf::ContextSettings	_glsettings;
	sf::RenderWindow	_win;

	std::mutex		_rmtx;

	std::vector <Shape>	_objects;
public:
	Simulation(const std::string & = "Sim");

	void add_shape(const Shape &);

	void run();

	// Static variables
	static size_t def_aa_level;
};

// Shape class
class Shape {
protected:
	// Center of mass (for physics)
	Vec3f	_com;

	// Position is not always the same thing for shapes
	// (ie. it could be the center or one of the corners).
	Vec3f	_pos;
public:
	Shape(const Vec3f &);
	Shape(const Vec3f &, const Vec3f &);

	virtual bool in(const Vec3f &) const = 0;
	virtual image::Color color(const Vec3f &) const = 0;
};

// Rect class
class Rect : public Shape {
	// Position is the geometric center

	// w, h, l
	Vec3f		_dim;
	image::Color	_col;

	// cached bounds (min, max)
	struct {
		float xmin;
		float xmax;

		float ymin;
		float ymax;

		float zmin;
		float zmax;
	} _bounds;
public:
	Rect(const Vec3f &, const Vec3f &,
		const image::Color & = image::WHITE);

	bool in(const Vec3f &) const override;
	image::Color color(const Vec3f &) const override;
};

int main()
{
	Simulation sim;

	Rect rect(Vec3f {0, 100, 200}, Vec3f {100, 100, 100});

	sim.run();
}

// Ray
Ray::Ray(const Vec3f &start, const Vec3f &delta)
		: _start(start), _delta(delta.normalized()) {}

Vec3f Ray::at(float t)
{
	return _start + t * _delta;
}

// Simulation
size_t Simulation::def_aa_level = 10;

Simulation::Simulation(const std::string &name)
		: _screen(800, 600, 4), _name(name),
		_glsettings(0, 0, def_aa_level),
		_win(sf::VideoMode {800, 600}, _name,
		sf::Style::Titlebar | sf::Style::Close, _glsettings) {}

void Simulation::add_shape(const Shape &shape)
{
	_objects.push_back(shape);
}

void Simulation::run()
{
	sf::Image image = _screen.sfml_image();

	sf::Texture texture;
	sf::Sprite sprite;

	while (_win.isOpen()) {
		sf::Event event;
		while (_win.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				_win.close();
		}

		texture.loadFromImage(image);
		sprite.setTexture(texture);

		_win.clear(sf::Color::Black);
		_win.draw(sprite);
		_win.display();
	}
}

// Shape
Shape::Shape(const Vec3f &pos) : _pos(pos) {}

Shape::Shape(const Vec3f &pos, const Vec3f &com) : _pos(pos), _com(com) {}

// Rect
Rect::Rect(const Vec3f &pos, const Vec3f &dim, const image::Color &col)
		: Shape(pos), _dim(dim), _col(col)
{
	_bounds.xmin = pos.x - dim.x/2;
	_bounds.xmax = pos.x + dim.x/2;

	_bounds.ymin = pos.y - dim.y/2;
	_bounds.ymax = pos.y + dim.y/2;

	_bounds.zmin = pos.z - dim.z/2;
	_bounds.zmax = pos.z + dim.z/2;
}

bool Rect::in(const Vec3f &pos) const
{
	return (pos.x >= _bounds.xmin && pos.x <= _bounds.xmax)
		&& (pos.y >= _bounds.ymin && pos.y <= _bounds.ymax)
		&& (pos.z >= _bounds.zmin && pos.z <= _bounds.zmax);
}

image::Color Rect::color(const Vec3f &pos) const
{
	if (in(pos))
		return _col;

	return image::BLACK;
}
