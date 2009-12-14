__kernel void threshold_byte(
	__global uchar* a,
	int width,
	int height,
	int x,
	int y,
	int ir_width,
	int ir_height,
	uchar value, uchar lessValue, uchar moreValue)
{
	int i;
	for(i = 0; i != width*height; ++i)
		a[i] = (a[i] > value) ? moreValue : lessValue;
}

__kernel void linearCombination_byte(
	__global uchar* v0,
	int width0,
	int height0,
	int x0,
	int y0,
	int ir_width0,
	int ir_height0,

	__global uchar* v1,
	int width1,
	int height1,
	int x1,
	int y1,
	int ir_width1,
	int ir_height1,

	float a,
	float b)
{
}

// v0 -> (gauss) -> v1
__kernel void gauss_byte(
	__global uchar* v0,
	int width0,
	int height0,
	int x0,
	int y0,
	int ir_width0,
	int ir_height0,

	__global uchar* v1,
	int width1,
	int height1,
	int x1,
	int y1,
	int ir_width1,
	int ir_height1,

	float sigma,
	uint n)
{
}


__kernel void threshold_float(
	__global float* a,
	int width,
	int height,
	int x,
	int y,
	int ir_width,
	int ir_height,
	float value, float lessValue, float moreValue)
{
	int i;
	for(i = 0; i != width*height; ++i)
		a[i] = (a[i] > value) ? moreValue : lessValue;
}

__kernel void linearCombination_float(
	__global float* v0,
	int width0,
	int height0,
	int x0,
	int y0,
	int ir_width0,
	int ir_height0,

	__global float* v1,
	int width1,
	int height1,
	int x1,
	int y1,
	int ir_width1,
	int ir_height1,

	float a,
	float b)
{
	
}


// v0 -> (gauss) -> v1
__kernel void gauss_float(
	__global float* v0,
	int width0,
	int height0,
	int x0,
	int y0,
	int ir_width0,
	int ir_height0,

	__global float* v1,
	int width1,
	int height1,
	int x1,
	int y1,
	int ir_width1,
	int ir_height1,

	float sigma,
	int n)
{

}


