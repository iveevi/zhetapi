  
CC	:= g++
CC	+= -std=c++11

TARGET	:= zhetapi

SRC	:= main.cpp

all: $(SRC)
	@echo BUILDING...
	@$(CC) -o $(TARGET) $(SRC)
	@echo DONE