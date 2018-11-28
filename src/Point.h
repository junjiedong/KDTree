/**
 * File: Point.h
 * -------------
 * A class representing a point in N-dimensional space.
 * Point is parameterized over an integer N.
 * This allows the compiler to verify the type being used correctly.
 */
#ifndef POINT_INCLUDED
#define POINT_INCLUDED

#include <cmath>
#include <algorithm>

template <std::size_t N>
class Point {
public:

    // Types representing iterators that can traverse and optionally modify the elements of the Point.
    typedef double* iterator;
    typedef const double* const_iterator;

    // Returns N, the dimension of the point.
    std::size_t size() const;

    // Queries or retrieves the value of the point at a particular point. The index is assumed to be in-range.
    double& operator[](std::size_t index);
    double operator[](std::size_t index) const;

    // Returns iterators delineating the full range of elements in the Point.
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

private:
    double coords[N];
};

// Returns the Squared Euclidean distance between two points.
template <std::size_t N>
double Distance(const Point<N>& one, const Point<N>& two);

// Returns whether two points are equal / not equal
template <std::size_t N>
bool operator==(const Point<N>& one, const Point<N>& two);

template <std::size_t N>
bool operator!=(const Point<N>& one, const Point<N>& two);

template <std::size_t N>
std::size_t Point<N>::size() const {
    return N;
}

template <std::size_t N>
double& Point<N>::operator[] (std::size_t index) {
    return coords[index];
}

template <std::size_t N>
double Point<N>::operator[] (std::size_t index) const {
    return coords[index];
}

template <std::size_t N>
typename Point<N>::iterator Point<N>::begin() {
    return coords;
}

template <std::size_t N>
typename Point<N>::const_iterator Point<N>::begin() const {
    return coords;
}

template <std::size_t N>
typename Point<N>::iterator Point<N>::end() {
    return begin() + size();
}

template <std::size_t N>
typename Point<N>::const_iterator Point<N>::end() const {
    return begin() + size();
}

template <std::size_t N>
double Distance(const Point<N>& one, const Point<N>& two) {
    double result = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        result += (one[i] - two[i]) * (one[i] - two[i]);
    return result;
}

template <std::size_t N>
bool operator==(const Point<N>& one, const Point<N>& two) {
    return std::equal(one.begin(), one.end(), two.begin());
}

template <std::size_t N>
bool operator!=(const Point<N>& one, const Point<N>& two) {
    return !(one == two);
}

#endif // POINT_INCLUDED
