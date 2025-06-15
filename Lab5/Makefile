CC = g++

CFLAGS = -Wall -g `pkg-config --cflags opencv4` -O0 

LDFLAGS = `pkg-config --libs opencv4`

all : main

main : main.o
	$(CC) -o main main.o $(LDFLAGS)

main.o : main.cpp main.h
	$(CC) $(CFLAGS) -c -o main.o main.cpp

viewer : viewer.o
	$(CC) -o viewer viewer.o $(LDFLAGS)

viewer.o : viewer.cpp
	$(CC) $(CFLAGS) -c -o viewer.o viewer.cpp

clean :
	rm *.o *~
