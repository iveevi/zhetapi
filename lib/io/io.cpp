#include "io.hpp"

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT_SYMBOL(printf, zhp_printf);

	ZHETAPI_EXPORT_SYMBOL(fprint, zhp_fprint);
	ZHETAPI_EXPORT_SYMBOL(fprintln, zhp_fprintln);
}
