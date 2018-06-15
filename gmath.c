#include "gmath.h"

//From http://developer.download.nvidia.com/cg/isinf.html
int isinf_gd (double s) {
  // By IEEE 754 rule, 2*Inf equals Inf
  return (2*s == s) && (s != 0);
}

//From http://developer.download.nvidia.com/cg/isnan.html
int isnan_gd (double s) {
  // By IEEE 754 rule, NaN is not equal to NaN
  return s != s;
}
