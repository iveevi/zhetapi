#ifndef RAW_TYPES_H_
#define RAW_TYPES_H_

// Engine headers
#include "../rational.hpp"
#include "../complex.hpp"
#include "../matrix.hpp"
#include "../vector.hpp"

namespace zhetapi {

// All valid types (use less vague naming, like Integer, Real, Boolean, etc)
using Z = long long int;
using Q = Rational <Z>;
using R = long double;

using B = bool;
using S = std::string;			

using CmpZ = Complex <Z>;
using CmpQ = Complex <Q>;
using CmpR = Complex <R>;			

using VecZ = Vector <Z>;
using VecQ = Vector <Q>;
using VecR = Vector <R>;
using VecCmpZ = Vector <CmpZ>;
using VecCmpQ = Vector <CmpQ>;
using VecCmpR = Vector <CmpR>;

using MatZ = Matrix <Z>;
using MatQ = Matrix <Q>;
using MatR = Matrix <R>;
using MatCmpZ = Matrix <CmpZ>;
using MatCmpQ = Matrix <CmpQ>;
using MatCmpR = Matrix <CmpR>;

}

#endif
