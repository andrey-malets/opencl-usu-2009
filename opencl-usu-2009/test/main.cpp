#include "library.h"
#include <iostream>

using opencl_usu_2009::ByteID;
using opencl_usu_2009::FloatID;
using opencl_usu_2009::UintID;

int tests();

template<typename IDType, typename Pixel> void testThreshold()
{
	std::cout << "start" << std::endl;
	size_t width = 7000, height = 7000, size = width*height;
		Pixel *data = new Pixel[size];
	std::cout << "allocated" << std::endl;
	for(size_t i = 0; i != size; ++i)
		data[i] = i;

	std::cout << "initialized" << std::endl;
	IDType obj(data, width, height);
	std::cout << "loaded" << std::endl;
	obj.trheshold(500000, 0, 1);
	std::cout << "processed" << std::endl;
	obj.unload(data);
	std::cout << "unloaded" << std::endl;

	for(size_t i = 0; i != size; ++i)
		if(i <= 500000)
		{
			if(data[i] != 0)
				std::cout << i << std::endl;
		}
		else
			if(data[i] != 1)
				std::cout << i << std::endl;

	std::cout << "checked" << std::endl;
	delete [] data;
}

//template<typename Data> double times(Data &functor)
//{
//	clock_t c = clock();
//	functor();
//	return (clock() - c) * 1. / CLOCKS_PER_SEC;
//}
//
//template<typename Data> class Runner
//{
//public:
//	Runner(Data &value) : value(value), n(n) { }
//	void operator()() { for(size_t i = 0; i != size; ++i) value(); }
//private:
//	Data value;
//	size_t n;
//};
//
//template<typename Type> class OpenCLThresholdTest
//{
//public:
//	OpenCLThresholdTest(size_t width, size_t height) { }
//
//private:
//
//};
//

int main(int argc, const char** argv)
{
//	testThreshold<UintID, UintID::Data>();
//	testThreshold<opencl_usu_2009::FloatID, float>();
	tests();
}
