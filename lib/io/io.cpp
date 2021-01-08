#include "io.hpp"

ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT_SYMBOL(printf, __zhp_std_printf);

	ZHETAPI_EXPORT_SYMBOL(fprint, __zhp_std_fprint);
	ZHETAPI_EXPORT_SYMBOL(fprintln, __zhp_std_fprintln);
}
