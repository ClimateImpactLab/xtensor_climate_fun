#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <math.h>
#include <iostream>
#include <numeric>
#include <cmath>

namespace py = pybind11;

// Examples

inline double example1(xt::pyarray<double> &m)
{
    return m(0);
}

inline xt::pyarray<double> example2(xt::pyarray<double> &m)
{
    return m + 2;
}

// Readme Examples

inline double readme_example1(xt::pyarray<double> &m)
{
    auto sines = xt::sin(m);
    return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
}

// Vectorize Examples

inline double scalar_func(double i, double j)
{
    return std::sin(i) + std::cos(j);
}

// Snyder EDD
inline xt::pyarray<double> snyder_edd(xt::pyarray<double> &tasmin, xt::pyarray<double> &tasmax, double threshold)
{
    // compute useful quantities for use in the transformation
    xt::pyarray<double> snyder_mean = ((tasmax + tasmin) / 2.0);
    xt::pyarray<double> snyder_width = ((tasmax - tasmin) / 2.0);
    xt::pyarray<double> snyder_theta = xt::asin( (threshold - snyder_mean) / snyder_width );

    // the trasnformation is computed using numpy arrays, taking advantage of
    // numpy's second where clause. Note that in the current dev build of
    // xarray, xr.where allows this functionality. As soon as this goes live,
    // this block can be replaced with xarray
    xt::pyarray<double> res = xt::where(
        tasmin < threshold,
        xt::where(
            tasmax > threshold,
            ((snyder_mean - threshold) * (M_PI / 2.0 - snyder_theta)
                + (snyder_width * xt::cos(snyder_theta))) / M_PI,
            0.0),
        snyder_mean - threshold);
    
    // wrap data in xarray DataArray, with the same dimensions and coordinates
    // as tasmin.

    return res;
}


// Python Module and Docstrings

PYBIND11_PLUGIN(xtensor_climate_fun)
{
    xt::import_numpy();

    py::module m("xtensor_climate_fun", R"docu(
        Fun with xtensor data!

        .. currentmodule:: xtensor_climate_fun

        .. autosummary::
           :toctree: _generate

           example1
           example2
           readme_example1
           vectorize_example1
    )docu");

    m.def("example1", example1, "Return the first element of an array, of dimension at least one");
    m.def("example2", example2, "Return the the specified array plus 2");

    m.def("snyder_edd", snyder_edd, "Return the snyder EDD");

    m.def("readme_example1", readme_example1, "Accumulate the sines of all the values of the specified array");

    m.def("vectorize_example1", xt::pyvectorize(scalar_func), "Add the sine and and cosine of the two specified values");

    return m.ptr();
}
