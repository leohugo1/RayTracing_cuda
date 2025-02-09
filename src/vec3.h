#pragma once
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <iostream>

 class Vec3 {
public:
	double e[3];
	__host__ __device__ Vec3() :e{ 0,0,0 } {};
	__host__ __device__ Vec3(double e0,double e1,double e2) :e{e0,e1, e2} {};
	__host__ __device__ inline double x()
	{
		return e[0];
	}
	__host__ __device__ inline double y()
	{
		return e[1];
	}
	__host__ __device__ inline double z()
	{
		return e[2];
	}
	__host__ __device__ inline Vec3 operator-() const 
	{
		return Vec3(-e[0], -e[1], -e[2]);
	}
	__host__ __device__ inline const Vec3& operator+() const
	{
		return *this;
	}
	__host__ __device__ inline double operator[](int i) const 
	{
		return e[i];
	}
	__host__ __device__ inline double& operator[](int i)
	{
		return e[i];
	}
	__host__ __device__ inline Vec3& operator+=(const Vec3& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}
	__host__ __device__ inline Vec3& operator-=(const Vec3& v)
	{
		e[0] -= v.e[0];
		e[1] -= v.e[1];
		e[2] -= v.e[2];
		return *this;
	}
	__host__ __device__ inline Vec3& operator*=(const Vec3& v)
	{
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}
	__host__ __device__ inline Vec3& operator/=(const Vec3& v)
	{
		e[0] /= v.e[0];
		e[1] /= v.e[1];
		e[2] /= v.e[2];
		return *this;
	}
	__host__ __device__ inline Vec3& operator*=(double t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}
	__host__ __device__ inline Vec3& operator/=(double t)
	{
		return *this *= 1 / t;
	}
	__host__ __device__ inline double length() const
	{
		return std::sqrt(length_squared());
	}
	__host__ __device__ inline double length_squared() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
	__host__ __device__ inline void make_unit_vector()
	{
		float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
		e[0] *= k; e[1] *= k; e[2] *= k;;
	}
	__host__ __device__ inline bool near_zero()
	{
		auto s = 1e-8;
		return ((fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s));
	}
};

 __host__ __device__ inline Vec3  operator+(const Vec3& v1, const Vec3& v2)
 {
	 return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
 }

 __host__ __device__ inline Vec3 operator-(const Vec3& v1, const Vec3& v2)
 {
	 return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
 }

 __host__ __device__ inline Vec3 operator*(const Vec3& v1, const Vec3& v2)
 {
	 return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
 }

 __host__ __device__ inline Vec3 operator/(const Vec3& v1, const Vec3& v2)
 {
	 return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
 }

 __host__ __device__ inline Vec3 operator*(float t, const Vec3& v)
 {
	 return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
 }

 __host__ __device__ inline Vec3 operator*(const Vec3& v, float t)
 {
	 return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
 }

 __host__ __device__ inline Vec3 operator/(Vec3 v, float t)
 {
	 return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
 }

 __host__ __device__ inline double dot(const Vec3& v1, const Vec3& v2)
 {
	 return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
 }

 __host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2)
 {
	 return Vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		 (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
		 (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
 }

 __host__ __device__ inline Vec3 unit_vector(Vec3 v)
 {
	 return v / v.length();
 }

 __device__ inline Vec3 random_double(curandState* state, double min, double max) {
	 double x = (min + (max - min) * curand_uniform(state));
	 double y = (min + (max - min) * curand_uniform(state));
	 double z = (min + (max - min) * curand_uniform(state));
	 return Vec3(x, y, z);
 }
 __device__ inline Vec3 random_unit_vector(curandState* state)
 {
	 Vec3 p;
	 double lensq;
	 int attempts = 0;
	 do {
		 p = random_double(state, -1, 1);
		 lensq = p.length_squared();
		 attempts++;
		 if (attempts > 10) return Vec3(1, 0, 0);
	 } while (lensq >= 1.0 || lensq < 1e-6);

	 return p / sqrt(fmax(lensq, 1e-6));
 }

 __device__ inline Vec3 random_on_hmisphere(const Vec3& normal, curandState* rand_state)
 {
	 Vec3 on_unit_sphere = random_unit_vector(rand_state);
	 if (dot(on_unit_sphere, normal) > 0.0)
		 return on_unit_sphere;
	 else
		 return -on_unit_sphere;
 }

 __device__ inline Vec3 reflect(const Vec3& v,const Vec3& n)
 {
	 return v - 2 * dot(v, n) * n;
 }

 __device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, double eta_over_etat)
 {
	 auto cos = fmin(dot(-uv, n), 1.0);
	 Vec3 r_out_perp = eta_over_etat * (uv + cos * n);
	 Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	 return r_out_perp + r_out_parallel;
 }

 __device__ inline Vec3 random_in_unit_disk(curandState* state)
 {
	 Vec3 p ;
	 do {
		 p = Vec3( (-1 + (1 - ( - 1)) * curand_uniform(state)),
		  (-1 + (1 - (-1)) * curand_uniform(state)),0);
	 } while (p.length_squared() > 1);
	 return p;
 }

 __device__ inline double random(curandState* state)
 {
	 return  (curand_uniform_double(state));
 }