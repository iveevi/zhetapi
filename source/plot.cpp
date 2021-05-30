#include <plot.hpp>

namespace zhetapi {

// Static variables
double Plot::zoom_factor = 1.25;
size_t Plot::def_width = 800;
size_t Plot::def_height = 600;
size_t Plot::def_aa_level = 10;
size_t Plot::def_plot_num = 1;
sf::Color Plot::def_axis_col = sf::Color(100, 100, 100);

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

void Plot::plot(const Vector <double> &point)
{
	sf::CircleShape circle;

	circle.setRadius(5);
	circle.setPosition(
		point[0] + _origin[0],
		_origin[1] - point[1]
	);

	_points.push_back({point, circle});
}

void Plot::plot(const std::function <double (double)> &ftn)
{
	sf::VertexArray curve(sf::LinesStrip, _width);

	for (int i = 0; i < _width; i++) {
		double x = i - _origin[0];

		Vector <double> p {
			x + _origin[0],
			_origin[1] - ftn(x)
		};

		curve[i].position = sf::Vector2f(p[0], p[1]);
	}

	_curves.push_back({ftn, curve});
}

void Plot::zoom(double factor)
{
	sf::View view = _win.getView();
	view.zoom(factor);
	_win.setView(view);
}

void Plot::show()
{
	_display = true;
	_win.setActive(false);
	_wman = new std::thread(&Plot::run, this);
}

void Plot::close()
{
	_display = false;
	_wman->join();
	delete _wman;
}

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

void Plot::init_axes()
{
	_axes.xscale = 10.0/_width;
	_axes.yscale = 10.0/_height;

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

}
