#include "constants.h"

/* Adapted from TVector code by igkiou */

namespace cuda {

template <typename T> struct TVector2 {
	T x, y;

	const static int dim = 2;

	__device__ TVector2() : x(0), y(0) {  }

	__device__ TVector2(T x, T y) : x(x), y(y) {  }

	__device__ explicit TVector2(T val) : x(val), y(val) { }

	__device__ template <typename T2> explicit TVector2(const TVector2<T2> &v)
		: x((T) v.x), y((T) v.y) { }

	__device__ template <typename T2> explicit TVector2(const T2 *v, const int size) {
		ASSERT(size == dim);
		x = static_cast<T>(v[0]);
		y = static_cast<T>(v[1]);
	}

	__device__ TVector2 operator+(const TVector2 &v) const {
		return TVector2(x + v.x, y + v.y);
	}

	__device__ TVector2 operator-(const TVector2 &v) const {
		return TVector2(x - v.x, y - v.y);
	}

	__device__ TVector2& operator+=(const TVector2 &v) {
		x += v.x; y += v.y;
		return *this;
	}

	__device__ TVector2& operator-=(const TVector2 &v) {
		x -= v.x; y -= v.y;
		return *this;
	}

	__device__ TVector2 operator*(T f) const {
		return TVector2(x*f, y*f);
	}

	__device__ TVector2 &operator*=(T f) {
		x *= f; y *= f;
		return *this;
	}

	__device__ TVector2 operator-() const {
		return TVector2(-x, -y);
	}

	__device__ TVector2 operator/(T f) const {
		ASSERT(abs(static_cast<Float>(f)) > M_EPSILON);
		T recip = (T) 1 / f;
		return TVector2(x * recip, y * recip);
	}

	__device__ TVector2 &operator/=(T f) {
		ASSERT(abs(static_cast<Float>(f)) > M_EPSILON);
		T recip = (T) 1 / f;
		x *= recip; y *= recip;
		return *this;
	}

	__device__ T &operator[](int i) {
		return (&x)[i];
	}

	__device__ T operator[](int i) const {
		return (&x)[i];
	}

	__device__ T lengthSquared() const {
		return x*x + y*y;
	}

	__device__ T min() const {
		return min(x, y);
	}

	__device__ T max() const {
		return max(x, y);
	}

	__device__ T length() const {
		return sqrt(lengthSquared());
	}

	__device__ void normalize() {
		if (!isZero()) {
			T tmp = length();
			x = x / tmp;
			y = y / tmp;
		}
	}

	__device__ void zero() {
		x = 0;
		y = 0;
	}

	__device__ bool isZero() const {
		return x == 0 && y == 0;
	}

	__device__ bool operator==(const TVector2 &v) const {
		return (v.x == x && v.y == y);
	}

	__device__ bool aproxEqual(const TVector2 &v) const {
		return (abs(v.x - x) < M_EPSILON * max((T) 1, max(abs(v.x), abs(x)))) && \
				(abs(v.y - y) < M_EPSILON * max((T) 1, max(abs(v.y), abs(y))));
	}

	__device__ bool operator!=(const TVector2 &v) const {
		return v.x != x || v.y != y;
	}

};

template <typename T>
__device__ inline TVector2<T> operator*(T f, const TVector2<T> &v) {
	return v*f;
}

template <typename T>
__device__ inline T dot(const TVector2<T> &v1, const TVector2<T> &v2) {
	return v1.x * v2.x + v1.y * v2.y;
}

template <typename T>
__device__ inline T absDot(const TVector2<T> &v1, const TVector2<T> &v2) {
	return abs(dot(v1, v2));
}


template <typename T>
__device__ inline TVector2<T> cross(const TVector2<T> &v1, const TVector2<T> & /* v2 */) {
	/* Left-handed vector cross product */
	return TVector2<T>(v1.y, - v1.x);
}

template <typename T>
__device__ inline TVector2<T> normalize(const TVector2<T> &v) {
	return v / v.length();
}

template <>
__device__ inline TVector2<int> TVector2<int>::operator/(int s) const {
	ASSERT(abs(static_cast<Float>(s)) > M_EPSILON);
	return TVector2(x/s, y/s);
}

template <>
__device__ inline TVector2<int> &TVector2<int>::operator/=(int s) {
	ASSERT(abs(static_cast<Float>(s)) > M_EPSILON);
	x /= s;
	y /= s;
	return *this;
}


// Adapted from Mitsuba's TVector3 class
template <typename T> struct TVector3 {
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
		ASSERT(size == dim);
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
		ASSERT(abs(static_cast<Float>(f)) > M_EPSILON);
		T recip = (T) 1 / f;
		return TVector3(x * recip, y * recip, z * recip);
	}

	/// Divide the vector by the given scalar
	__device__ TVector3 &operator/=(T f) {
		ASSERT(abs(static_cast<Float>(f)) > M_EPSILON);
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
		return sqrt(lengthSquared());
	}

	/// Return the min entry of this vector
	__device__ T min() const {
		return min(x, min(y, z));
	}

	/// Return the max entry of this vector
	__device__ T max() const {
		return max(x, max(y, z));
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
		return (abs(v.x - x) < M_EPSILON * max((T) 1, max(abs(v.x), abs(x)))) && \
				(abs(v.y - y) < M_EPSILON * max((T) 1, max(abs(v.y), abs(y)))) && \
				(abs(v.z - z) < M_EPSILON * max((T) 1, max(abs(v.z), abs(z))));
	}

	/// Inequality test
	__device__ bool operator!=(const TVector3 &v) const {
		return v.x != x || v.y != y || v.z != z;
	}

};

}
