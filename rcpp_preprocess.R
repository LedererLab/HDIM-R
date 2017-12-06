library(Rcpp)
Rcpp::compileAttributes("./HDIM")

library(devtools)
devtools::document("./HDIM")
