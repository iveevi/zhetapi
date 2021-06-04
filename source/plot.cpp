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
		_origin({_width/2.0f, _height/2.0f})
{
	def_plot_num++;

	_axes.xmin = -10.5;
	// Give some space for the axis ticks
	_axes.xmax = 10.5;

	_axes.ymin = -10.5;
	_axes.ymax = 10.5;

	_axes.color = def_axis_col;

	// TODO: use guaranteed fonts only
	_axes.font.loadFromFile("/usr/share/fonts/truetype/ubuntu/UbuntuMono-BI.ttf");

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
	sf::Event event;

	sf::Vector2i mstart;
	sf::Vector2i mend;

	bool dragging = false;

	while (_win.isOpen()) {
		if (!_display) {
			_win.close();

			break;
		}

		while (_win.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				_win.close();

			if (event.type == sf::Event::MouseWheelMoved)
				zoom(pow(zoom_factor, event.mouseWheel.delta));

			if (event.type == sf::Event::MouseButtonPressed) {
				mstart = sf::Mouse::getPosition(_win);
				dragging = true;

				// Stop early
				break;
			}

			if (event.type == sf::Event::MouseButtonReleased) {
				dragging = false;

				// Stop early
				break;
			}

			if (dragging) {
				mend = sf::Mouse::getPosition(_win);

				sf::Vector2i dpos = mend - mstart;

				double dx = (_axes.xmax - _axes.xmin) * (double) dpos.x/_width;
				double dy = (_axes.ymax - _axes.ymin) * (double) dpos.y/_height;

				_axes.xmax -= dx;
				_axes.xmin -= dx;

				_axes.ymax += dy;
				_axes.ymin += dy;

				_origin += {dpos.x, dpos.y};

				redraw();

				mstart = sf::Mouse::getPosition(_win);
			}
		}

		_win.clear(sf::Color::Black);

		// Axes
		_win.draw(_axes.x);
		_win.draw(_axes.y);

		// Draw arrows in a way that is not too tight
		if (_origin[0] > 0)
			_win.draw(_axes.xleft);

		if (_origin[0] < _width)
			_win.draw(_axes.xright);
		
		if (_origin[1] > 0)
			_win.draw(_axes.yup);
		
		if (_origin[1] < _height)
			_win.draw(_axes.ydown);

		for (const Curve &curve : _curves)
			_win.draw(curve.va);

		for (const Point &point : _points)
			_win.draw(point.circle);

		_win.draw(_axes.xtext);
		_win.draw(_axes.ytext);

		_win.display();
	}
}

// TODO: make private and rename to set_axis
void Plot::init_axes()
{
	// x-axis
	float cx = std::max(std::min(_origin[1], (float) _height - 5.0f), 5.0f);

	_axes.x = sf::VertexArray(sf::LinesStrip, 2);
	_axes.x[0].position = {0, cx};
	_axes.x[1].position = {_width, cx};
	_axes.x[0].color = _axes.color;
	_axes.x[1].color = _axes.color;

	// y-axis
	float cy = std::max(std::min(_origin[0], (float) _width - 5.0f), 5.0f);
	
	_axes.y = sf::VertexArray(sf::LinesStrip, 2);
	_axes.y[0].position = {cy, 0};
	_axes.y[1].position = {cy, _height};
	_axes.y[0].color = _axes.color;
	_axes.y[1].color = _axes.color;

	// Axis arrows
	_axes.xleft.setPointCount(3);
	_axes.xright.setPointCount(3);
	_axes.yup.setPointCount(3);
	_axes.ydown.setPointCount(3);

	_axes.xleft.setFillColor(_axes.color);
	_axes.xright.setFillColor(_axes.color);
	_axes.yup.setFillColor(_axes.color);
	_axes.ydown.setFillColor(_axes.color);

	_axes.xleft.setPoint(0, {10, cx + 5});
	_axes.xleft.setPoint(1, {10, cx - 5});
	_axes.xleft.setPoint(2, {0, cx});

	_axes.xright.setPoint(0, {_width - 10, cx + 5});
	_axes.xright.setPoint(1, {_width - 10, cx - 5});
	_axes.xright.setPoint(2, {_width, cx});

	_axes.yup.setPoint(0, {cy + 5, 10});
	_axes.yup.setPoint(1, {cy - 5, 10});
	_axes.yup.setPoint(2, {cy, 0});

	_axes.ydown.setPoint(0, {cy + 5, _height - 10});
	_axes.ydown.setPoint(1, {cy - 5, _height - 10});
	_axes.ydown.setPoint(2, {cy, _height});

	// Axis labels
	_axes.xtext.setFont(_axes.font);
	_axes.ytext.setFont(_axes.font);

	_axes.xtext.setCharacterSize(25);
	_axes.ytext.setCharacterSize(25);

	_axes.xtext.setString("x");
	_axes.xtext.setFillColor(_axes.color);

	_axes.ytext.setString("y");
	_axes.ytext.setFillColor(_axes.color);

	// Labels
	float xlabelx = (_origin[0] > _width/2) ? 20.0f : (_width - 20.0f);
	float xlabely = (_origin[1] >= _height/2) ? (cx - 40.0f) : cx;
	
	float ylabelx = (_origin[0] > _width/2) ? (cy - 20.0f) : cy + 10.0f;
	float ylabely = (_origin[1] >= _height/2) ? 5.0f : (_height - 40.0f);

	_axes.xtext.setPosition({xlabelx, xlabely});
	_axes.ytext.setPosition({ylabelx, ylabely});
}

// TODO: Rename
void Plot::redraw()
{
	init_axes();

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

// Private methods
/* TODO: any use?
bool Plot::out_of_bounds_x() const
{
	return (_origin[0] < 0 || _origin[0] > _width);
}

// TODO: any use?
bool Plot::out_of_bounds_y() const
{
	return (_origin[1] < 0 || _origin[1] > _height);
}

// TODO: any use?
bool Plot::out_of_bounds() const
{
	return out_of_bounds_x() || out_of_bounds_y();
} */

}
