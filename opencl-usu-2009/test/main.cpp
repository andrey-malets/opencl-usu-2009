#include "library.h"

int main(int argc, const char** argv)
{
	byte *image = new byte[100*100];
	opencl_usu_2009::Identificator<byte> id(image, 100, 100), id2(id);
	delete[] image;
}
