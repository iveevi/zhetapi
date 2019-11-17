CC	:= g++

TRG	:= calc

SRC	:= main.cpp

FLG	:= -std=c++17 -g -O -Wall -Weffc++ -pedantic  \
-pedantic-errors -Wextra -Waggregate-return -Wcast-align \
-Wcast-qual -Wconversion \
-Wdisabled-optimization \
-Werror -Wfloat-equal -Wformat=2 \
-Wformat-nonliteral -Wformat-security  \
-Wformat-y2k \
-Wimplicit  -Wimport  -Winit-self  -Winline \
-Winvalid-pch   \
-Wlong-long \
-Wmissing-field-initializers -Wmissing-format-attribute   \
-Wmissing-include-dirs -Wmissing-noreturn \
-Wpacked  -Wpadded -Wpointer-arith \
-Wredundant-decls \
-Wshadow -Wstack-protector \
-Wstrict-aliasing=2 -Wswitch-default \
-Wswitch-enum \
-Wunreachable-code -Wunused \
-Wunused-parameter \
-Wvariadic-macros \
-Wwrite-strings

make:
	@echo BUILDING...
	@$(CC) $(FLG) $(SRC) -o $(TRG)
	@echo RUNNING...

	@if test -s ./calc;\
		then ./$(TRG);\
	else\
		echo BUILD FAILURE; \
	fi