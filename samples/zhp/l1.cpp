ZHETAPI_LIBRARY()
{
	ZHETAPI_EXPORT(print_hello_world);
	ZHETAPI_EXPORT_SYMBOL(hello, print_hello_world);
	ZHETAPI_EXPORT_SYMBOL(first, get_first);
}