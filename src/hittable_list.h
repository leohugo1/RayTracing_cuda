#pragma once
#include "hittable.h"
#include "aabb.h"

class HittableList: public Hittable{
public:
	Hittable** objects;
	int size;
	AABB bbox;
	__device__ HittableList() {};
	__device__ HittableList(Hittable** list, int size) : objects(list), size(size) {};
	__device__ void add(Hittable* obj,int i)
	{
		objects[i] = obj;
		bbox = AABB(bbox, obj->bounding_box());
	}
	__device__ bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& rec)const override
	{
		HitRecord temp_rec;
		bool hit_anything = false;
		auto close_so_far = ray_tmax;

		for (int i = 0; i < size; i++)
		{
			if (objects[i]->hit(r, ray_tmin, close_so_far, temp_rec))
			{
				hit_anything = true;
				close_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}

	__device__ AABB bounding_box() const override { return bbox; }
};