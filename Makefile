INCLUDES = -I/usr/local/include/eigen3
CFLAGS = -std=c++14 -Ofast
all: multi-layer-train.cpp
	g++ $(CFLAGS) -o multi-train multi-layer-train.cpp $(INCLUDES)
	g++ $(CFLAGS) -o multi-test multi-layer-test.cpp $(INCLUDES)

clean:
	$(RM) multi-train
	$(RM) multi-test
	cd Weights; $(RM) *