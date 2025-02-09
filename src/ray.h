#pragma once
#include "cuda_runtime.h"
#include "vec3.h"

 class Ray {

public:

	__device__ Ray() {};
	//__device__ Ray(const Vec3& origin, const Vec3& direction) :Ray(origin,direction,0) {};
	__device__ Ray(const Vec3& origin, const Vec3& direction,double time) :orig(origin), dir(direction),tm(time) {};
	__device__ Ray(const Vec3& origin, const Vec3& direction) :orig(origin), dir(direction) {};
	__device__ const Vec3& origin() const
	{
		return orig;
	}
	__device__ const Vec3& direction() const
	{
		return dir;
	}
	__device__ Vec3 at(double t) const
	{
		return orig + t * dir;
	}

	__device__ double time() const
	{
		return tm;
	}

private:
	Vec3 orig;
	Vec3 dir;
	double tm = 0;
};
