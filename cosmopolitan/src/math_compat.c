#include <limits.h>
#include <math.h>

#if !defined(__COSMOPOLITAN__)
#error "This compatibility unit is specific to Cosmopolitan 4.0.2"
#endif

_Static_assert(sizeof(long) == sizeof(long long), "Cosmopolitan must use LP64");
_Static_assert(LONG_MIN == LLONG_MIN && LONG_MAX == LLONG_MAX,
               "The rounding result types must have identical ranges");

/* The pinned SDK declares these functions but only supplies lround/lroundl.
   Equal result ranges let those native functions preserve rounding and fenv
   behavior. This unit is compiled with -fno-builtin to retain these calls. */
long long llround(double value) {
    return lround(value);
}

long long llroundl(long double value) {
    return lroundl(value);
}
