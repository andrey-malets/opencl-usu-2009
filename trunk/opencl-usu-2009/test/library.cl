#define M_PI       3.14159265358979323846

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

	float a, float b)
{
	int i = get_global_id(0);
	if(i <= ir_width0*ir_height0)
	{
		int sindex = x1 + y1 * width1 + i % ir_width1 + width1 * (i / ir_width1);
		int dindex = x0 + y0 * width0 + i % ir_width0 + width0 * (i / ir_width0);

		float res = a*v0[dindex] + b*v1[sindex];
		if(res > UCHAR_MAX)
			v0[dindex] = UCHAR_MAX;
		else
			if(res < 0)
				v0[dindex] = 0;
			else
				v0[dindex] = res;
	}
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
	uint n,
	__global float *w)
{
	int myId = get_global_id(0);
	if(myId == 0)
	{
		float s = 0.0;
		for(int i = 0; i != 2*n+1; ++i)
			for(int j = 0; j != 2*n+1; ++j)
				s+= w[i*(2*n+1)+j] =
					1/(2 * M_PI * sigma * sigma)
					* exp ((float) -((i-n-1)*(i-n-1)+(j-n-1)*(j-n-1))
					/ (2 * sigma * sigma));
		for(int i = 0; i != (2*n+1)*(2*n+1); ++i)
			w[i] /= s;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(myId <= ir_width1*ir_height1)
	{
		int dindex = x1 + y1 * width1 + myId % ir_width1 + width1 * (myId / ir_width1);
		float res = 0;
		for(int i = 0; i != (2*n+1)*(2*n+1); ++i)
		{
			int sindex = x0 + y0* width0 + (myId + i - n - 1) % ir_width0 + width0 * ((myId + i - n - 1) / ir_width0);
			res += v0[sindex] * w[i];
		}

		if(res > UCHAR_MAX)
			v1[dindex] = UCHAR_MAX;
		else
			if(res < 0)
				v1[dindex] = 0;
			else
				v1[dindex] = res;
	}
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

	float a, float b)
{
	int i = get_global_id(0);
	if(i <= ir_width0*ir_height0)
	{
		int sindex = x1 + y1 * width1 + i % ir_width1 + width1 * (i / ir_width1);
		int dindex = x0 + y0 * width0 + i % ir_width0 + width0 * (i / ir_width0);
		v0[dindex] = a*v0[dindex] + b*v1[sindex];
	}
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
	uint n,
	__global float *w)
{
	int myId = get_global_id(0);
	if(myId == 0)
	{
		float s = 0.0;
		for(int i = 0; i != 2*n+1; ++i)
			for(int j = 0; j != 2*n+1; ++j)
				s+= w[i*(2*n+1)+j] =
					1/(2 * M_PI * sigma * sigma)
					* exp ((float) -((i-n-1)*(i-n-1)+(j-n-1)*(j-n-1))
					/ (2 * sigma * sigma));
		for(int i = 0; i != (2*n+1)*(2*n+1); ++i)
			w[i] /= s;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(myId <= ir_width1*ir_height1)
	{
		int dindex = x1 + y1 * width1 + myId % ir_width1 + width1 * (myId / ir_width1);
		v1[dindex] = 0;
		for(int i = -n; i != n+1; ++i)
			for(int j = -n; j != n+1; ++j)
			{
				int sindex = x0 + i + (y0 + j)* width0 + myId % ir_width0 + width0 * (myId / ir_width0);
				v1[dindex] += v0[sindex] * w[(i+n)*(2*n+1)+(j+n)];
			}
	}
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

	float a, float b)
{
	int i = get_global_id(0);
	if(i <= ir_width0*ir_height0)
	{
		int sindex = x1 + y1 * width1 + i % ir_width1 + width1 * (i / ir_width1);
		int dindex = x0 + y0 * width0 + i % ir_width0 + width0 * (i / ir_width0);

		float res = a*v0[dindex] + b*v1[sindex];
		if(res > UINT_MAX)
			v0[dindex] = UINT_MAX;
		else
			if(res < 0)
				v0[dindex] = 0;
			else
				v0[dindex] = res;
	}
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
	uint n,
	__global float *w)
{
	int myId = get_global_id(0);
	if(myId == 0)
	{
		float s = 0.0;
		for(int i = 0; i != 2*n+1; ++i)
			for(int j = 0; j != 2*n+1; ++j)
				s+= w[i*(2*n+1)+j] =
					1/(2 * M_PI * sigma * sigma)
					* exp ((float) -((i-n-1)*(i-n-1)+(j-n-1)*(j-n-1))
					/ (2 * sigma * sigma));
		for(int i = 0; i != (2*n+1)*(2*n+1); ++i)
			w[i] /= s;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	if(myId <= ir_width1*ir_height1)
	{
		int dindex = x1 + y1 * width1 + myId % ir_width1 + width1 * (myId / ir_width1);
		float res = 0;
		for(int i = -n; i != n+1; ++i)
			for(int j = -n; j != n+1; ++j)
			{
				int sindex = x0 + i + (y0 + j)* width0 + myId % ir_width0 + width0 * (myId / ir_width0);
				res += v0[sindex] * w[(i+n)*(2*n+1)+(j+n)];
			}

		if(res > UINT_MAX)
			v0[dindex] = UINT_MAX;
		else
			if(res < 0)
				v0[dindex] = 0;
			else
				v0[dindex] = res;
	}
}
