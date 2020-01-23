  
CC	:= g++
CC	+= -std=c++11

TARGET	:= zhetapi

SRC	:= main.cpp

all: $(SRC)
	@echo BUILDING...
	@$(CC) -o $(TARGET) -DDEBUG=0 -g $(SRC)
	@echo DONE
