#include "ray.h"
#include "hittable.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

class Material {
public:
    __device__ virtual bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* rand_state) const = 0;
};

class Lambertian : public Material {
public:
    __device__ Lambertian() {};
    __device__ Lambertian(const Vec3& a) :albedo(a) {};

    __device__  bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,curandState* rand_state) const override
    {
        auto scatter_direction =  rec.normal + random_unit_vector(rand_state);
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;
        scattered = Ray(rec.p, scatter_direction,r.time());
        attenuation = albedo;
        return true;
    }
public:
    Vec3 albedo;
};


class Metal :public Material {
public:
   __device__ Metal(const Vec3& a,double fuzz) :albedo(a),fuzz(fuzz<1?fuzz:1) {};
   __device__ bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* rand_state) const override
   {
       Vec3 reflected = reflect(r.direction(), rec.normal);
       reflected = unit_vector(reflected) + (fuzz * random_unit_vector(rand_state));
       scattered = Ray(rec.p, reflected,r.time());
       attenuation = albedo;

       return (dot(scattered.direction(),rec.normal)> 0);
   }

private:
    Vec3 albedo;
    double fuzz;
};

class Dielectric :public Material {
public:
    __device__ Dielectric(double refraction_index) :refraction_index(refraction_index) {};
    __device__ bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* rand_state) const override
    {
        attenuation = Vec3(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;
        Vec3 unit_direction = unit_vector(r.direction());
        double cos_theta = fminf(dot(-unit_direction, rec.normal), 1);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        bool cannot_refract = ri * sin_theta > 1.0;
        Vec3 direction;
        if (cannot_refract)
        {
            direction = reflect(unit_direction, rec.normal);
        }
        else
        {
            direction = refract(unit_direction, rec.normal, ri);
        }
        scattered = Ray(rec.p, direction,r.time());
        return true;
    }

    double refraction_index;
};