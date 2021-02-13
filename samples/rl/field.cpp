#include "global.hpp"

// Force field
Vec F(const Vec &x)
{
	return Vec {
		0.5 * x.y(),
		-2.0 * x.x()
	};
};

// Average field strength
double Fc(const Vec &s, double r)
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
