CC	:= g++

TRG	:= calc

SRC	:= main.cpp

FLG	:= -std=c++17

make:
	@$(CC) $(FLG) $(SRC) -o $(TRG)

	@if test -s ./calc;\
		then ./$(TRG);\
	else\
		echo BUILD FAILURE; \
	fi