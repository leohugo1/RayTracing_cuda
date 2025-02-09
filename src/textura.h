#pragma once
#include "cuda_runtime.h"
#include "vec3.h"


class Texture {
public:
	__device__ virtual Vec3 value(double u, double v, Vec3& p) const = 0;
};

class Solid_color :public Texture {
public:
	__device__ Solid_color(const Vec3& albedo) :albedo(albedo) {};
	__device__ Solid_color(double red, double green, double blue) :Solid_color(Vec3(red, green, blue)) {};

	__device__ Vec3 value(double u, double v, Vec3& p) const override
	{
		return albedo;
	}

	Vec3 albedo;
};