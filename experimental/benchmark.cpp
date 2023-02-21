// Benchmark headers
#include <benchmark/benchmark.h>

// Library headers
#include "../include/autograd/autograd.hpp"
#include "../include/autograd/ml.hpp"

using namespace zhetapi;
using namespace zhetapi::autograd;

// Performance of a single dense layer
static void dense_layer(benchmark::State &state)
{
	Variable x;
	Function model = ml::dense(1000, 2000)(x);
	Constant in(
		{2000},
		[](size_t i) {
			return 1.0f;
		}
	);

	for (auto _ : state)
		model(in);
}

BENCHMARK(dense_layer)->Unit(benchmark::kMillisecond);

// Performance of a deep dense network
static void dense_network(benchmark::State &state)
{
	Variable x;
	Function model = ml::dense(1000, 2000)(x);
	model = ml::dense(2000, 2000)(model);
	model = ml::dense(2000, 1000)(model);

	Constant in(
		{2000},
		[](size_t i) {
			return 1.0f;
		}
	);

	for (auto _ : state)
		model(in);
}

BENCHMARK(dense_network)->Unit(benchmark::kMillisecond);

// Benchmarking matrix multiplication
template <class T>
Matrix <T> simple_fma(const Matrix <T> &a, const Matrix <T> &b, const Matrix <T> &c)
{
	Matrix <T> out = a * b;
	out += c;

	return out;
}

template <class T>
void inline_fma(T *out, const T *matrix, const T *bias, const T *input, size_t rows, size_t cols)
{
	for (size_t i = 0; i < rows; i++) {
		T sum = 0;

		for (size_t j = 0; j < cols; j++)
			sum += matrix[i * cols + j] * input[j];

		out[i] = sum + bias[i];
	}
}

template <class T>
void parallel_fma(T *out, const T *matrix, const T *bias, const T *input, size_t rows, size_t cols)
{
#pragma omp parallel for
	for (long int i = 0; i < rows; i++) {
		T sum = 0;

		const T *c = &matrix[i * cols];
		for (size_t j = 0; j < cols; j++)
			sum += c[i] * input[j];

		out[i] = sum + bias[i];
	}
}

static void matrix_multiply(benchmark::State &state)
{
	Matrix <float> w1 {1000, 2000, [](size_t i) { return 1.0f; }};
	Matrix <float> b1 {1000, 1, [](size_t i) { return 1.0f; }};

	Matrix <float> in {2000, 1, [](size_t i) { return 1.0f; }};

	for (auto _ : state)
		simple_fma(w1, in, b1);
}

BENCHMARK(matrix_multiply)->Unit(benchmark::kMillisecond);

static void matrix_multiply_inline_fma(benchmark::State &state)
{
	Matrix <float> w1 {1000, 2000, [](size_t i) { return 1.0f; }};
	Matrix <float> b1 {1000, 1, [](size_t i) { return 1.0f; }};

	Matrix <float> in {2000, 1, [](size_t i) { return 1.0f; }};
	Matrix <float> out {1000, 1, [](size_t i) { return 0.0f; }};

	for (auto _ : state)
		inline_fma(out.data(), w1.data(), b1.data(), in.data(), 1000, 1000);
}

BENCHMARK(matrix_multiply_inline_fma)->Unit(benchmark::kMillisecond);

static void matrix_multiply_parallel_fma(benchmark::State &state)
{
	Matrix <float> w1 {1000, 2000, [](size_t i) { return 1.0f; }};
	Matrix <float> b1 {1000, 1, [](size_t i) { return 1.0f; }};

	Matrix <float> in {2000, 1, [](size_t i) { return 1.0f; }};
	Matrix <float> out {1000, 1, [](size_t i) { return 0.0f; }};

	for (auto _ : state)
		parallel_fma(out.data(), w1.data(), b1.data(), in.data(), 1000, 1000);
}

BENCHMARK(matrix_multiply_parallel_fma)->Unit(benchmark::kMillisecond);

// Main
BENCHMARK_MAIN();
