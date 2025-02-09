#pragma once
#include <iostream>
#include "cuda_runtime.h"

class Interval {
public:
	double min, max;
	__device__ Interval():min(+INFINITY), max(-INFINITY) {};
	__device__ Interval(double min, double max) :min(min), max(max) {};
	__device__ Interval(const Interval& a, const Interval& b)
	{
		min = a.min <= b.min ? a.min : b.min;
		max = a.max >= b.max ? a.max : b.max;
	}

	__device__ double size() const
	{
		return max - min;
	}
	__device__ bool constains(double x) const
	{
		return min <= x && x <= max;
	}
	__device__ bool surrounds(double x) const
	{
		return min < x && x < max;
	}

	__device__ double clamp(double x) const
	{
		if (x < min) return min;
		if (x > max) return max;
		return x;
	}
	__device__ Interval expand(double delta) const
	{
		auto padding = delta / 2;
		return Interval(min - padding, max + padding);
	}

	static const Interval empty, universe;
};