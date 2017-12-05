#!/bin/bash

#Let Rcpp update ./HDIM/R/RcppExports.R and ./HDIM/src/RcppExports.R
Rscript ./rcpp_preprocess.R

R CMD INSTALL --preclean --no-multiarch --with-keep.source HDIM
