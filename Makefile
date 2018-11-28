CPPFLAGS=-std=c++14
OPFLAGS=-O3

all:
	g++ ${CPPFLAGS} ${OPFLAGS} -o main src/main.cc

clean:
	rm -f main
