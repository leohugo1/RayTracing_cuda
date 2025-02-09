#pragma once
#include "vec3.h"
#include "ray.h"
#include "aabb.h"

class Material;

class HitRecord {
public:
	Vec3 p;
	Vec3 normal;
	double t;
	bool front_face;
	Material* material;
	double u;
	double v;

	__device__ void setFrontFace(const Ray& r, const Vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class Hittable {
public:
	__device__ virtual bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& rec) const = 0;
	__device__ virtual AABB bounding_box() const = 0;
};