__kernel void threshold_byte(__global uchar* a, int size, uchar value, uchar lessValue, uchar moreValue)
{
	int i;
	for(i = 0; i != size; ++i);
		a[i] = (a[i] > value) ? moreValue : lessValue;
}

__kernel void gauss_byte(__global const float* a) { }

__kernel void linearCombination_byte(__global const float* a) { }



__kernel void threshold_float(__global const float* a) { }

__kernel void gauss_float(__global const float* a) { }

__kernel void linearCombination_float(__global const float* a) { }
