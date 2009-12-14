#include "library.h"
#include <iostream>

int tests();

void testThresholdByte()
{
	const size_t size = 10*10;

	byte *image = new byte[size], *ref = new byte[size];
	for(size_t i = 0; i != size; ++i)
		image[i] = i;

	opencl_usu_2009::ByteID id(image, 10, 10);

	id.trheshold(50, 0, 1);

	id.unload(ref);

	for(size_t i = 0; i != size; ++i)
		if(i <= 50)
		{
			if(ref[i] != 0)
				std::cout << i << std::endl;
		}
		else
			if(ref[i] != 1)
				std::cout << i << std::endl;

	delete[] image;
	delete[] ref;
}

void testThresholdFloat()
{
	const size_t size = 10*10;

	float *image = new float[size], *ref = new float[size];
	for(size_t i = 0; i != size; ++i)
		image[i] = i;

	opencl_usu_2009::FloatID id(image, 10, 10);
	id.trheshold(50, 0.0, 1.0);
	id.unload(ref);

	for(size_t i = 0; i != size; ++i)
		if(i <= 50)
		{
			if(ref[i] != 0.0)
				std::cout << i << std::endl;
		}
		else
			if(ref[i] != 1.0)
				std::cout << i << std::endl;

	delete[] image;
	delete[] ref;
}

int main(int argc, const char** argv)
{
//	testThresholdByte();
//	testThresholdFloat();
	tests();
}

