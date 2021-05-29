#ifndef PLOT_H_
#define PLOT_H_

// C/C++ headers
#include <thread>
#include <vector>

// Engine headers
#include <vector.hpp>

// Graphics headers
#include <SFML/Window.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics.hpp>

namespace zhetapi {

class Plot {
	std::string			_name;

	size_t				_width = def_width;
	size_t				_height = def_height;

	std::thread *			_wman = nullptr;

	std::vector <sf::VertexArray>	_curves;
	std::vector <sf::CircleShape>	_points;

	// Coordinate axes
	struct {
		// Make as arrays next time
		sf::VertexArray x;
		sf::VertexArray y;

		sf::ConvexShape xleft;
		sf::ConvexShape xright;
		sf::ConvexShape yup;
		sf::ConvexShape ydown;

		sf::Color color;
	} _axes;

	// Origin
	Vector <double>			_origin;

	sf::ContextSettings		_glsettings;
	sf::RenderWindow		_win;

	bool				_display = false;
public:
	Plot(const std::string & = "Plot #" + std::to_string(def_plot_num));

	void plot(const Vector <double> &);
	void plot(const std::function <double (double)> &);

	void zoom(double);

	void show();
	void close();

	void run();

	void init_axes();

	static size_t def_width;
	static size_t def_height;
	static size_t def_aa_level;
	static size_t def_plot_num;
	static sf::Color def_axis_col;
};

}

#endif
