#include "vec3.h"

namespace Util {
    inline std::istream& operator>>(std::istream& is, Vec3& t) {
        is >> t.e[0] >> t.e[1] >> t.e[2];
        return is;
    }

    inline std::ostream& operator<<(std::ostream& os, const Vec3& t) {
        os << t.e[0] << " " << t.e[1] << " " << t.e[2];
        return os;
    }
    
}