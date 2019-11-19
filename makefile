CC	:= g++

TRG	:= calc

SRC	:= main.cpp

FLG	:= -std=c++17 -g

make:
	@echo BUILDING...
	@$(CC) $(FLG) $(SRC) -o $(TRG)
	@echo RUNNING...

	@if test -s ./calc;\
		then ./$(TRG);\
	else\
		echo BUILD FAILURE; \
	fi