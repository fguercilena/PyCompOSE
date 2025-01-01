#include <cmath>
#include <cassert>
#include <cstdlib>

using namespace std;


template<typename T>
auto sgn(const T &x) {
  return (T(0) < x) - (x < T(0));
}

int64_t as_int(_Float64 f) {
  return *reinterpret_cast<int64_t*>(&f);
}

_Float64 as_double(int64_t i) {
  return *reinterpret_cast<_Float64*>(&i);
}

extern "C" _Float64 NQT_log_LANL_single(const _Float64 x) {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
  constexpr _Float64 scale_down = 2.220446049250313e-16;
  return static_cast<_Float64>(as_int(x) - one_as_int) * scale_down;
}

extern "C" _Float64 NQT_exp_LANL_single(const _Float64 x) {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // as_int(2.0) - as_int(1.0)
  constexpr _Float64 scale_up = 4503599627370496;
  return as_double(static_cast<int64_t>(x*scale_up) + one_as_int);
}

extern "C" _Float64 NQT_log_frexp_O1_single(const _Float64 x) {
  int32_t exponent;
  assert(x > 0 && "log divergent for x <= 0");
  const _Float64 y = frexp(x, &exponent);
  _Float64 mantissa = 2*y-1;
  return mantissa + exponent -1;
}

extern "C" _Float64 NQT_exp_ldexp_O1_single(const _Float64 x) {
  const int32_t flr = std::floor(x);
  const _Float64 remainder = x - flr;
  const _Float64 mantissa = 0.5 * (remainder + 1);
  const int32_t exponent = flr + 1;
  return ldexp(mantissa, exponent);
}



extern "C" _Float64 NQT_log_frexp_O2_single(const _Float64 x) {
  int32_t exponent;
  assert(x > 0 && "log divergent for x <= 0");
  const _Float64 y = frexp(x, &exponent);
  _Float64 mantissa = 2*y-1;
  return mantissa*(4-mantissa)/3 + exponent - 1;
}

extern "C" _Float64 NQT_exp_ldexp_O2_single(const _Float64 x) {
  const int32_t flr = std::floor(x);
  const _Float64 remainder = x - flr;
  const _Float64 tmp = 2-std::sqrt(4-3*remainder);
  const _Float64 mantissa = 0.5 * (tmp + 1);
  const int32_t exponent = flr + 1;
  return ldexp(mantissa, exponent);
}



extern "C" _Float64 NQT_log_O2_single(const _Float64 x) {
      // as_int(1.0) == 2^62 - 2^52
      constexpr int64_t one_as_int = 4607182418800017408;
      // 1/(as_int(2.0) - as_int(1.0)) == 2^-52
      constexpr _Float64 scale_down = 2.220446049250313e-16;
      // 2^52 - 1
      constexpr int64_t mantissa_mask = 4503599627370495;
      // 2^26 - 1
      constexpr int64_t low_mask = 67108863;

      const int64_t x_as_int = as_int(x) - one_as_int;
      const int64_t frac_as_int = x_as_int & mantissa_mask;
      const int64_t frac_high = frac_as_int>>26;
      const int64_t frac_low  = frac_as_int & low_mask;
      const int64_t frac_squared = frac_high*frac_high + ((frac_high*frac_low)>>25);

      return static_cast<_Float64>(x_as_int +
                                    ((frac_as_int - frac_squared)/3)) * scale_down;
  }


extern "C" _Float64 NQT_exp_O2_single(const _Float64 x) {
      // as_int(1.0) == 2^62 - 2^52
      constexpr int64_t one_as_int = 4607182418800017408;
      // as_int(2.0) - as_int(1.0) == 2^52
      constexpr _Float64 scale_up = 4503599627370496;
      constexpr int64_t mantissa_mask = 4503599627370495; // 2^52 - 1
      constexpr int64_t a = 9007199254740992; // 2 * 2^52
      constexpr _Float64 b = 67108864; // 2^26
      constexpr int64_t c = 18014398509481984; // 4 * 2^52

      const int64_t x_as_int = static_cast<int64_t>(x*scale_up);
      const int64_t frac_as_int = x_as_int & mantissa_mask;
      const int64_t frac_sqrt = static_cast<int64_t>(
                                  b*std::sqrt(static_cast<_Float64>(c-3*frac_as_int)));

      return as_double(x_as_int + a - frac_sqrt - frac_as_int + one_as_int);
  }

extern "C" void NQT_log_LANL_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_log_LANL_single(x[idx]);
  }
  return;
}

extern "C" void NQT_exp_LANL_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_exp_LANL_single(x[idx]);
  }
  return;
}

extern "C" void NQT_log_frexp_O1_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_log_frexp_O1_single(x[idx]);
  }
  return;
}

extern "C" void NQT_exp_ldexp_O1_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_exp_ldexp_O1_single(x[idx]);
  }
  return;
}


extern "C" void NQT_log_frexp_O2_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_log_frexp_O2_single(x[idx]);
  }
  return;
}

extern "C" void NQT_exp_ldexp_O2_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_exp_ldexp_O2_single(x[idx]);
  }
  return;
}


extern "C" void NQT_log_O2_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_log_O2_single(x[idx]);
  }
  return;
}

extern "C" void NQT_exp_O2_array(const _Float64* x, _Float64* f, const int32_t count) {
  for (int32_t idx=0; idx<count; ++idx) {
    f[idx] = NQT_exp_O2_single(x[idx]);
  }
  return;
}
