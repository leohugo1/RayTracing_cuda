#pragma once
#include "hittable.h"
#include "hittable_list.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


class BVH_NODE :public Hittable
{
public:
	__device__ BVH_NODE(Hittable** objects, size_t start, size_t end,curandState* state) 
	{
		int axis = int(3 * curand_uniform(state));

		auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

		size_t object_span = end - start;

		if (object_span == 1)
		{
			left = right = objects[start];
		}
		else if (object_span == 2)
		{
			if (comparator(objects[start], objects[start + 1])) {
				left = objects[start];
				right = objects[start + 1];
			}
			else {
				left = objects[start + 1];
				right = objects[start];
			}
		}else
		{
			thrust::device_ptr<Hittable*> dev_ptr(objects + start);
			thrust::sort(thrust::device, dev_ptr, dev_ptr + object_span, comparator);
			size_t mid = start + object_span / 2;
			left = new BVH_NODE(objects, start, mid, state);
			right = new BVH_NODE(objects, mid, end, state);
		}

		bbox = AABB(left->bounding_box(), right->bounding_box());
	};

	__device__ bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& rec) const override
	{
		if (!bbox.hit(r, Interval(ray_tmin, ray_tmax))) return false;

		bool hit_left = left->hit(r, ray_tmin, ray_tmax, rec);
		bool hit_right = right->hit(r, ray_tmin, hit_left?rec.t:ray_tmax, rec);

		return hit_left || hit_right;
	}

	__device__ AABB bounding_box() const override
	{
		return bbox;
	}
private:
	__device__ static bool box_compare(const Hittable* a, const Hittable* b, int axis_index)
	{
		auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
		auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
		return a_axis_interval.min < b_axis_interval.min;
	}

	__device__ static bool box_x_compare(const Hittable* a, const Hittable* b)
	{
		return box_compare(a, b, 0);
	}

	__device__ static bool box_y_compare(const Hittable* a, const Hittable* b)
	{
		return box_compare(a, b, 1);
	}

	__device__ static bool box_z_compare(const Hittable* a, const Hittable* b)
	{
		return box_compare(a, b, 2);
	}
private:
	Hittable* left;
	Hittable* right;
	AABB bbox;
};