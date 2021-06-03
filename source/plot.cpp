#include <plot.hpp>

namespace zhetapi {

// Static variables
double Plot::zoom_factor = 1.25;
size_t Plot::def_width = 800;
size_t Plot::def_height = 600;
size_t Plot::def_aa_level = 10;
size_t Plot::def_plot_num = 1;
sf::Color Plot::def_axis_col = sf::Color(100, 100, 100);

/**
 * @brief Name constructor. Initializes the center of the canvas to the origin.
 * The bounds of the x and y axis are both initialized to [-10.5, 10.5].
 * Defaults the size of the canvas to 800 x 600 pixels.
 * 
 * @param name the name of the plot (also the plot title). Defaults to "Plot
 * #{pnum}," where pnum is a static, incremented plot number.
 */
Plot::Plot(const std::string &name)
		: _name(name), _glsettings(0, 0, def_aa_level),
		_win(sf::VideoMode {_width, _height}, _name,
		sf::Style::Titlebar | sf::Style::Close, _glsettings),
		_origin({_width/2.0, _height/2.0})
{
	def_plot_num++;

	// Axes
	_axes.color = def_axis_col;
	init_axes();
}

/**
 * @brief Plots a point. Note that the point will not be visible if the
 * coordinate is not in bounds of the axes. 
 * 
 * @param coords the coordinate of the point to be plotted. 
 */
void Plot::plot(const Vector <double> &coords)
{
	sf::CircleShape circle;

	circle.setRadius(5);

	sf::Vector2f tc = true_coords(coords);
	circle.setPosition({
		tc.x - 5,
		tc.y - 5
	});

	_points.push_back({coords, circle});
}

/**
 * @brief Plots a function. The function is evaluated for each pixel on the
 * width of the canvas (so if the width of the canvas is 800 pixels then there
 * would be 800 distinct evaulateions of the function). The value of the
 * function is appropriately scaled.
 * 
 * @param ftn the function to be plotted.
 */
void Plot::plot(const std::function <double (double)> &ftn)
{
	sf::VertexArray curve(sf::LinesStrip, _width);

	// TODO: cache dx and dy
	double dx = _axes.xmax - _axes.xmin;
	for (int i = 0; i < _width; i++) {
		double x = _axes.xmin + (i * dx)/_width;

		curve[i].position = true_coords({x, ftn(x)});
	}

	_curves.push_back({ftn, curve});
}

// TODO: make private
void Plot::zoom(double factor)
{
	// TODO: take origin into consideration
	_axes.xmin /= factor;
	_axes.xmax /= factor;
	_axes.ymin /= factor;
	_axes.ymax /= factor;

	redraw();
}

/**
 * @brief Displays the plot on another thread. Any modifications to the plot
 * will be displayed in real time.
 */
void Plot::show()
{
	_display = true;
	_win.setActive(false);
	_wman = new std::thread(&Plot::run, this);
}

/**
 * @brief Closes the plot if it is not already closed.
 */
void Plot::close()
{
	// TODO: check if not already closed
	_display = false;
	_wman->join();
	delete _wman;
}

// TODO: make private
void Plot::run()
{
	while (_win.isOpen()) {
		if (!_display) {
			_win.close();

			break;
		}

		sf::Event event;

		while (_win.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				_win.close();
			if (event.type == sf::Event::MouseWheelMoved)
				zoom(pow(zoom_factor, event.mouseWheel.delta));
		}

		_win.clear(sf::Color::Black);

		// Axes
		_win.draw(_axes.x);
		_win.draw(_axes.y);
		_win.draw(_axes.xleft);
		_win.draw(_axes.xright);
		_win.draw(_axes.yup);
		_win.draw(_axes.ydown);

		for (const Curve &curve : _curves)
			_win.draw(curve.va);

		for (const Point &point : _points)
			_win.draw(point.circle);

		_win.display();
	}
}

// TODO: make private
void Plot::init_axes()
{
	// Give some space for the ticks
	_axes.xmin = -10.5;
	_axes.xmax = 10.5;
	
	_axes.ymin = -10.5;
	_axes.ymax = 10.5;

	_axes.x = sf::VertexArray(sf::LinesStrip, 2);
	_axes.x[0].position = {0, _origin[1]};
	_axes.x[1].position = {_width, _origin[1]};
	_axes.x[0].color = _axes.color;
	_axes.x[1].color = _axes.color;

	_axes.y = sf::VertexArray(sf::LinesStrip, 2);
	_axes.y[0].position = {_origin[0], 0};
	_axes.y[1].position = {_origin[0], _height};
	_axes.y[0].color = _axes.color;
	_axes.y[1].color = _axes.color;

	// Arrows
	_axes.xleft.setPointCount(3);
	_axes.xright.setPointCount(3);
	_axes.yup.setPointCount(3);
	_axes.ydown.setPointCount(3);

	_axes.xleft.setFillColor(_axes.color);
	_axes.xright.setFillColor(_axes.color);
	_axes.yup.setFillColor(_axes.color);
	_axes.ydown.setFillColor(_axes.color);

	_axes.xleft.setPoint(0, {10, _origin[1] + 5});
	_axes.xleft.setPoint(1, {10, _origin[1] - 5});
	_axes.xleft.setPoint(2, {0, _origin[1]});

	_axes.xright.setPoint(0, {_width - 10, _origin[1] + 5});
	_axes.xright.setPoint(1, {_width - 10, _origin[1] - 5});
	_axes.xright.setPoint(2, {_width, _origin[1]});

	_axes.yup.setPoint(0, {_origin[0] + 5, 10});
	_axes.yup.setPoint(1, {_origin[0] - 5, 10});
	_axes.yup.setPoint(2, {_origin[0], 0});

	_axes.ydown.setPoint(0, {_origin[0] + 5, _height - 10});
	_axes.ydown.setPoint(1, {_origin[0] - 5, _height - 10});
	_axes.ydown.setPoint(2, {_origin[0], _height});
}

void Plot::redraw()
{
	for (Point &point : _points) {
		sf::Vector2f tc = true_coords(point.coords);
		point.circle.setPosition({
			tc.x - 5,
			tc.y - 5
		});
	}

	double dx = _axes.xmax - _axes.xmin;
	for (Curve &curve : _curves) {
		for (int i = 0; i < _width; i++) {
			double x = _axes.xmin + (i * dx)/_width;

			curve.va[i].position = true_coords({x, curve.ftn(x)});
		}
	}
}

sf::Vector2f Plot::true_coords(const Vector <double> &coords)
{
	double i = _width * (coords[0] - _axes.xmin)/(_axes.xmax - _axes.xmin);
	double j = _height * (1 - (coords[1] - _axes.ymin)/(_axes.ymax - _axes.ymin));

	return {i, j};
}

}
