#pragma once
#include "hittable.h"
#include "vec3.h"

class Sphere :public Hittable {
public:
	__device__ Sphere(const Vec3& center, double radius,Material* material) :center(center,Vec3(0,0,0)), radius(radius), mat(material)
    {
        auto rvec = Vec3(radius, radius, radius);
        bbox = AABB(center - rvec, center + rvec);
    };
    __device__ Sphere(const Vec3& center1,const Vec3& center2 ,double radius, Material* material) :center(center1,center2 -center1), radius(radius), mat(material) 
    {
        auto rvec = Vec3(radius, radius, radius);
        AABB box1(center.at(0) - rvec, center.at(0) + rvec);
        AABB box2(center.at(1) - rvec, center.at(1) + rvec);
        bbox = AABB(box1, box2);
    };

	__device__ bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& rec) const override {
        Vec3 current_center = center.at(r.time());
        Vec3 oc = current_center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius * radius;
        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        auto root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward_normal = (rec.p - current_center) / radius;
        rec.setFrontFace(r, outward_normal);
        rec.material = mat;

        return true;
	}
    __device__ AABB bounding_box() const override { return bbox; };
private:
	Ray center;
	double radius;
    Material* mat;
    AABB bbox;
};