CPPFLAGS=-std=c++14
OPFLAGS=-O3

all:
	g++ ${CPPFLAGS} ${OPFLAGS} -O3 -o main src/main.cc

clean:
	rm -f main
