# HDIM-R

HDIM is a toolkit for working with high-dimensional data that emphasizes
speed and statistical guarantees. Specifically, it provides tools for working
with the LASSO objective function.

HDIM provides iterative solvers for the LASSO objective function
 including ISTA, FISTA and Coordinate Descent. HDIM also provides FOS,
  the Fast and Optimal Selection algorithm, a novel new method
for performing variable selection via the LASSO.

This package is a R wrapper over the original [C++ implementation of HDIM](https://github.com/LedererLab/FOS).

## Dependencies

This package requires that the [Eigen 3](http://eigen.tuxfamily.org/index.php)
C++ linear algebra package be installed on the target system. Note that the root
directory for Eigen 3, labeled `eigen3`, should be located under `/usr/include/`.

## Supported Platforms

This package has been tested on, and currently supports the **Linux** and **Windows**
 operating systems.

## Installation

### Windows

Installation under Windows *requires* that the root directory of the Eigen3 library
be located in the same directory as the root of the HDIM package.

##### Dependencies

* [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
* [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)

##### Building

- Clone the HDIM package into the directory where the Eigen3 library is located.
- Navigate to $PKG_DIR/R_Wrapper, where PKG_DIR is the root directory of the cloned repository.
- Find the file `win_build.ps1` and run it using PowerShell.
- This will run a preprocessing step using Rcpp then build and install the R Wrapper.


### Linux

##### Dependencies

* [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)

##### Building

- Clone the HDIM package into a convenient location.
- Navigate to $PKG_DIR/R_Wrapper, where PKG_DIR is the root directory of the cloned repository.
- Find the file `nix_build.sh` and mark it as executable ( `chmod +x ./nix_build.sh` ).
- This will run a preprocessing step using Rcpp then build and install the R Wrapper.

## Licensing

The HDIM-R package is licensed under the GPLv3 license. To view the GPLv3 license please consult
the `HDIM/GPLv3.txt` file included with this package.

## Authors

Based on research conducted by the Lederer and Hauser HDIM group including work
 by Johannes Lederer, Alain Hauser, Néhémy Lim, and others.

The original C++ implementation of the HDIM package was constructed by Saba Noorassa
 and Benjamin J Phillips.

This package was constructed by Benjamin J Phillips.

## References

* [FOS](https://arxiv.org/abs/1609.07195)
* [GAP SAFE Screening Rules](https://arxiv.org/abs/1505.03410)
