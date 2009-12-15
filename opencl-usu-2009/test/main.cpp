#include "library.h"
#include <iostream>

template<typename IDType, typename Pixel> void testThreshold()
{
	std::cout << "start" << std::endl;
	size_t width = 10000, height = 10000, size = width*height;
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

int main(int argc, const char** argv)
{
	testThreshold<opencl_usu_2009::UintID, cl_uint>();
//	testThreshold<opencl_usu_2009::FloatID, float>();
}
