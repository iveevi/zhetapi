#include "port.hpp"

TEST(matrix_construction_and_memory)
{
	using namespace zhetapi;

	Matrix <double> tmp;

	oss << "Default constructor: " << tmp << endl;

	return true;
}

TEST(kernel_apt_and_mult)
{
	using namespace zhetapi;
	using Mat = Matrix <double>;
	using Vec = Vector <double>;

	static const size_t rounds = 3;
	static const long double limit = 5;
	static const utility::Interval <> elemd(limit);

	for (size_t i = 0; i < rounds; i++) {
		// M is 4 x 5 and V is 4 x 1
		Mat M(4, 5,
			[](size_t i, size_t j) -> double {
				return elemd.uniform();
			}
		);

		Vec V(4,
			[](size_t i) -> double {
				return elemd.uniform();
			}
		);

		Vec out1 = Vec(M * V.append_above(1));
		Vec out2 = apt_and_mult(M, V);

		oss << "Outputs:" << std::endl;
		oss << "\tout1 = " << out1 << std::endl;
		oss << "\tout2 = " << out2 << std::endl;

		double error = (out1 - out2).norm();

		oss << "\terror = " << error << std::endl;

		if (error > 1e-10) {
			oss << "\t\tToo high!" << std::endl;
			return false;
		}
	}

	return true;
}

TEST(kernel_rmt_and_mult)
{
	using namespace zhetapi;
	using Mat = Matrix <double>;
	using Vec = Vector <double>;

	static const size_t rounds = 3;
	static const long double limit = 5;
	static const utility::Interval <> elemd(limit);

	for (size_t i = 0; i < rounds; i++) {
		// M is 4 x 5 and V is 4 x 1
		Mat M(4, 5,
			[](size_t i, size_t j) -> double {
				return elemd.uniform();
			}
		);

		Vec V(4,
			[](size_t i) -> double {
				return elemd.uniform();
			}
		);

		Vec out1 = Vec(M.transpose() * V).remove_top();
		Vec out2 = rmt_and_mult(M, V);

		oss << "Outputs:" << std::endl;
		oss << "\tout1 = " << out1 << std::endl;
		oss << "\tout2 = " << out2 << std::endl;

		double error = (out1 - out2).norm();

		oss << "\terror = " << error << std::endl;

		if (error > 1e-10) {
			oss << "\t\tToo high!" << std::endl;
			return false;
		}
	}

	return true;
}

TEST(kernel_vvt_mult)
{
	using namespace zhetapi;
	using Mat = Matrix <double>;
	using Vec = Vector <double>;

	static const size_t rounds = 3;
	static const long double limit = 5;
	static const utility::Interval <> elemd(limit);

	for (size_t i = 0; i < rounds; i++) {
		Vec v1(5,
			[](size_t i) -> double {
				return elemd.uniform();
			}
		);

		Vec v2(4,
			[](size_t i) -> double {
				return elemd.uniform();
			}
		);

		Mat out1 = v1 * v2.transpose();
		Mat out2 = vvt_mult(v1, v2);

		oss << "Outputs:" << std::endl;
		oss << "\tout1 = " << out1 << std::endl;
		oss << "\tout2 = " << out2 << std::endl;

		double error = (out1 - out2).norm();

		oss << "\terror = " << error << std::endl;

		if (error > 1e-10) {
			oss << "\t\tToo high!" << std::endl;
			return false;
		}
	}

	return true;
}