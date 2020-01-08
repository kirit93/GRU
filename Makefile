INCLUDES = -I/usr/local/include/eigen3
CFLAGS = -std=c++14 -Ofast
all: train.cpp
	g++ $(CFLAGS) -o train train.cpp $(INCLUDES)
	g++ $(CFLAGS) -o test test.cpp $(INCLUDES)

clean:
	$(RM) train
	$(RM) test