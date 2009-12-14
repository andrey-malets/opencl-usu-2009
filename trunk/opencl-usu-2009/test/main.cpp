#include "library.h"
#include <iostream>

int main(int argc, const char** argv)
{
	const size_t size = 10000*10000;

	byte *image = new byte[size], *ref = new byte[size];
	for(size_t i = 0; i != size; ++i)
		image[i] = i;

	opencl_usu_2009::Identificator<byte> id(image, 10000, 10000);

	id.trheshold(100, 100, 100);

	id.unload(ref);
/*
	for(size_t i = 0; i != size; ++i)
		if(image[i] != ref[i])
			std::cout << i << std::endl;
*/
	delete[] image;
}
