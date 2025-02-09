#pragma once
#include "ray.h"
#include <corecrt_math_defines.h>

class Camera {
public:
	__device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect,curandState* state)
	{
		Vec3 u, v, w;
		local_state = *state;
		float theta = vfov * M_PI / 180.0;
		float half_height = tan(theta / 2.0);
		float half_width = aspect * half_height;
		w = unit_vector( lookfrom - lookat);
		u = unit_vector(cross( vup,w));
		v = cross(w, u);
		origin = lookfrom;
		horizontal = half_width*u * focus_dist;
		vertical =  half_height * v;
		lower_left_corner = origin - horizontal/2 - vertical/2 - (focus_dist * w);
		auto angle_theta = (defocus_angle / 2) * M_PI / 180;
		auto defocus_radius = focus_dist * std::tan(angle_theta);
		defocus_disk_u = u * defocus_radius;
		defocus_disk_v = v * defocus_radius;
	}
	

	__device__ Ray getRay(float u, float v)
	{
		auto ray_origin = (defocus_angle <= 0) ? origin : defocus_disk_sample(&local_state);
		Vec3 dir = lower_left_corner + u * horizontal + v * vertical - ray_origin;
		auto ray_time = random(&local_state);
		return Ray(ray_origin,unit_vector(dir),ray_time);
	}

	__device__  Vec3 defocus_disk_sample(curandState* state) const {
		// Returns a random point in the camera defocus disk.
		auto p = random_in_unit_disk(state);
		return origin + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
	}

	Vec3 origin;
	Vec3 lower_left_corner;
	Vec3 horizontal;
	Vec3 vertical;
	Vec3 defocus_disk_u;       
	Vec3 defocus_disk_v;
	double defocus_angle = 0;  
	double focus_dist = 1;
	curandState local_state;
};