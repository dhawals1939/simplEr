#include "constants.h"

void Assert(bool p) {
    (void)p;
}

// Adapted from Mitsuba's TVector3 class
template <typename T> struct TVector3 {
	typedef T          value_type;
	T x, y, z;

	/// Number of dimensions
	const static int dim = 3;

	/// default constructor
	__device__ TVector3() : x(0), y(0), z(0) {  }

	/// Initialize the vector with the specified X, Y and Z components
	__device__ TVector3(T x, T y, T z) : x(x), y(y), z(z) {  }

	/// Initialize all components of the the vector with the specified value
	__device__ explicit TVector3(T val) : x(val), y(val), z(val) { }

	/// Initialize the vector with the components of another vector data structure
	template <typename T2> __device__ explicit TVector3(const TVector3<T2> &v)
		: x((T) v.x), y((T) v.y), z((T) v.z) { }

	/// Initialize from a C-style array. For use with MATLAB mxArrays with
	/// double data fields.
	// Not safe if incorrectly sized array is passed. size argument is used for
	// some weak safety checking.
	template <typename T2> __device__ explicit TVector3(const T2 *v, const int size) {
		Assert(size == dim);
		x = static_cast<T>(v[0]);
		y = static_cast<T>(v[1]);
		z = static_cast<T>(v[2]);
	}

	/// Add two vectors and return the result
	__device__ TVector3 operator+(const TVector3 &v) const {
		return TVector3(x + v.x, y + v.y, z + v.z);
	}

	/// Subtract two vectors and return the result
	__device__ TVector3 operator-(const TVector3 &v) const {
		return TVector3(x - v.x, y - v.y, z - v.z);
	}

	/// Add another vector to the current one
	__device__ TVector3& operator+=(const TVector3 &v) {
		x += v.x; y += v.y; z += v.z;
		return *this;
	}

	/// Subtract another vector from the current one
	__device__ TVector3& operator-=(const TVector3 &v) {
		x -= v.x; y -= v.y; z -= v.z;
		return *this;
	}

	/// Multiply the vector by the given scalar and return the result
	__device__ TVector3 operator*(T f) const {
		return TVector3(x*f, y*f, z*f);
	}

	/// Multiply the vector by the given scalar
	__device__ TVector3 &operator*=(T f) {
		x *= f; y *= f; z *= f;
		return *this;
	}

	/// Return a negated version of the vector
	__device__ TVector3 operator-() const {
		return TVector3(-x, -y, -z);
	}

	/// Divide the vector by the given scalar and return the result
	__device__ TVector3 operator/(T f) const {
		Assert(std::abs(static_cast<Float>(f)) > M_EPSILON);
		T recip = (T) 1 / f;
		return TVector3(x * recip, y * recip, z * recip);
	}

	/// Divide the vector by the given scalar
	__device__ TVector3 &operator/=(T f) {
		Assert(std::abs(static_cast<Float>(f)) > M_EPSILON);
		T recip = (T) 1 / f;
		x *= recip; y *= recip; z *= recip;
		return *this;
	}

	/// Index into the vector's components
	__device__ T &operator[](int i) {
		return (&x)[i];
	}

	/// Index into the vector's components (const version)
	__device__ T operator[](int i) const {
		return (&x)[i];
	}

	/// Return the squared 2-norm of this vector
	__device__ T lengthSquared() const {
		return x*x + y*y + z*z;
	}

	/// Return the 2-norm of this vector
	__device__ T length() const {
		return std::sqrt(lengthSquared());
	}

	/// Return the min entry of this vector
	__device__ T min() const {
		return std::min(x, std::min(y, z));
	}

	/// Return the max entry of this vector
	__device__ T max() const {
		return std::max(x, std::max(y, z));
	}

	__device__ void normalize() {
		if (!isZero()) {
			T tmp = length();
			x = x / tmp;
			y = y / tmp;
			z = z / tmp;
		}
	}

	__device__ void zero() {
		x = 0;
		y = 0;
		z = 0;
	}

	/// Return whether or not this vector is identically zero
	__device__ bool isZero() const {
		return x == 0 && y == 0 && z == 0;
	}

	/// Equality test
	__device__ bool operator==(const TVector3 &v) const {
		return (v.x == x && v.y == y && v.z == z);
	}

	/// Approximate equality test
	__device__ bool aproxEqual(const TVector3 &v) const {
		return (std::abs(v.x - x) < M_EPSILON * std::max((T) 1, std::max(std::abs(v.x), std::abs(x)))) && \
				(std::abs(v.y - y) < M_EPSILON * std::max((T) 1, std::max(std::abs(v.y), std::abs(y)))) && \
				(std::abs(v.z - z) < M_EPSILON * std::max((T) 1, std::max(std::abs(v.z), std::abs(z))));
	}

	/// Inequality test
	__device__ bool operator!=(const TVector3 &v) const {
		return v.x != x || v.y != y || v.z != z;
	}

};
