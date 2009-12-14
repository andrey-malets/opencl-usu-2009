__kernel void threshold_byte(
	__global uchar* a,
	uint width,
	uint height,
	uint x,
	uint y,
	uint ir_width,
	uint ir_height,
	uchar value, uchar lessValue, uchar moreValue)
{
	int i = get_global_id(0);
	if(i <= ir_width*ir_height)
	{
		int index = x + y * width + i % ir_width + width * (i / ir_width);
		a[index] = (a[index] > value) ? moreValue : lessValue;
	}
}

__kernel void linearCombination_byte(
	__global uchar* v0,
	uint width0,
	uint height0,
	uint x0,
	uint y0,
	uint ir_width0,
	uint ir_height0,

	__global uchar* v1,
	uint width1,
	uint height1,
	uint x1,
	uint y1,
	uint ir_width1,
	uint ir_height1,

	float a,
	float b)
{
}

// v0 -> (gauss) -> v1
__kernel void gauss_byte(
	__global uchar* v0,
	uint width0,
	uint height0,
	uint x0,
	uint y0,
	uint ir_width0,
	uint ir_height0,

	__global uchar* v1,
	uint width1,
	uint height1,
	uint x1,
	uint y1,
	uint ir_width1,
	uint ir_height1,

	float sigma,
	uint n)
{
}


__kernel void threshold_float(
	__global float* a,
	uint width,
	uint height,
	uint x,
	uint y,
	uint ir_width,
	uint ir_height,
	float value, float lessValue, float moreValue)
{
	int i = get_global_id(0);
	if(i <= ir_width*ir_height)
	{
		int index = x + y * width + i % ir_width + width * (i / ir_width);
		a[index] = (a[index] > value) ? moreValue : lessValue;
	}
}

__kernel void linearCombination_float(
	__global float* v0,
	uint width0,
	uint height0,
	uint x0,
	uint y0,
	uint ir_width0,
	uint ir_height0,

	__global float* v1,
	uint width1,
	uint height1,
	uint x1,
	uint y1,
	uint ir_width1,
	uint ir_height1,

	float a,
	float b)
{
	
}


// v0 -> (gauss) -> v1
__kernel void gauss_float(
	__global float* v0,
	uint width0,
	uint height0,
	uint x0,
	uint y0,
	uint ir_width0,
	uint ir_height0,

	__global float* v1,
	uint width1,
	uint height1,
	uint x1,
	uint y1,
	uint ir_width1,
	uint ir_height1,

	float sigma,
	uint n)
{

}


__kernel void threshold_uint(
	__global uint* a,
	uint width,
	uint height,
	uint x,
	uint y,
	uint ir_width,
	uint ir_height,
	uint value, uint lessValue, uint moreValue)
{
	int i = get_global_id(0);
	if(i <= ir_width*ir_height)
	{
		int index = x + y * width + i % ir_width + width * (i / ir_width);
		a[index] = (a[index] > value) ? moreValue : lessValue;
	}
}

__kernel void linearCombination_uint(
	__global float* v0,
	uint width0,
	uint height0,
	uint x0,
	uint y0,
	uint ir_width0,
	uint ir_height0,

	__global float* v1,
	uint width1,
	uint height1,
	uint x1,
	uint y1,
	uint ir_width1,
	uint ir_height1,

	float a,
	float b)
{
	
}


// v0 -> (gauss) -> v1
__kernel void gauss_uint(
	__global float* v0,
	uint width0,
	uint height0,
	uint x0,
	uint y0,
	uint ir_width0,
	uint ir_height0,

	__global float* v1,
	uint width1,
	uint height1,
	uint x1,
	uint y1,
	uint ir_width1,
	uint ir_height1,

	float sigma,
	uint n)
{

}


