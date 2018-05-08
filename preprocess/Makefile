all: gen_features_v1 split_val gen_features_v2 gen_features_v3 gen_features_v18

CXX = g++

CXXFLAGS =    -O3 -Wall -fmessage-length=0
#CXXFLAGS =    -g -Wall -fmessage-length=0
#CXXFLAGS += -fno-stack-protector
gen_features_v1: gen_features_v1.o
	$(CXX) -o $@ $^

gen_features_v2: gen_features_v2.o
	$(CXX) -o $@ $^

gen_features_v3: gen_features_v3.o
	$(CXX) -o $@ $^

gen_features_v18: gen_features_v18.o
	$(CXX) -o $@ $^

split_val: split_val.o
	$(CXX) -o $@ $^

clean:
	rm *.o -rf
	rm gen_features_v1 -rf

