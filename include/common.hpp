#ifndef ZHETAPI_COMMON_H_
#define ZHETAPI_COMMON_H_

// Printing vectors
template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector <T> &v)
{
	os << "{";
	for (size_t i = 0; i < v.size(); i++) {
		os << v[i];
		if (i != v.size() - 1)
			os << ", ";
	}

	return os << "}";
}

#endif
