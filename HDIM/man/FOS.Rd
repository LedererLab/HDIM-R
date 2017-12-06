\name{FOS}
\alias{FOS}
\title{Fast and Optimal Support Algorithm}
\description{
    Fit a generalized linear model via penalized maximum likelihood.  The
  regularization path is computed for the lasso or elasticnet penalty at a grid
  of values for the regularization parameter lambda. Can deal with all
  shapes of data, including very large sparse data matrices. Fits
  linear, logistic and multinomial, poisson, and Cox regression models.
  }
\usage{
FOS(X, Y, solver_type, use_single_precision=FALSE)
}

\arguments{
  \item{x}{An n x p design matrix.}
  \item{y}{A 1 x p array representing the predictors}
  \item{solver_type}{The type of iterative solver used internally. Can used
      sub-gradient descent methods or coordinate descent, both of which
      can use GAPSAFE screening rules or not.}
}
\details{
  Perform an L1 regularized linear regression using the Fast and Optimal Support
  algorithm.
  }
\value{
A list containing the results of the regression.
  \item{beta}{The coefficients of the regression results}
  \item{index}{The number of the grid element where the algorithm stops, in the range 1 - 100 }
  \item{lambda}{The regularization parameter that was deemed optimal}
  \item{intercept}{Value of the intercept term.}
  \item{support}{The estimated support. Enteries will be 0 if the regression coefficient was
  zero, and 1 if it was non-zero and signifigant.}
}
\references{Friedman, J., Hastie, T. and Tibshirani, R. (2008)
  \emph{Efficient Feature Selection With Large and High-dimensional Data},
     \url{https://arxiv.org/abs/1609.07195cr
  \emph{Pre-print via ArXiv}\cr
  \url{https://arxiv.org}\cr
  Néhémy Lim, Johannes Lederer (2016)
  }
\author{Benjamin J Phillips\cr
Maintainer: Benjamin J Phillips \email{bejphil@uw.edu}}
\seealso{\code{ISTA}, \code{FISTA}, and \code{CD} functions.}
\examples{
  library(HDIM)

  dataset <- matrix(rexp(200, rate=.1), ncol=20)

  yinput <- dataset[, 1, drop = FALSE]
  xinput <- dataset[, names(dataset) != names(yinput)]

  fos_fit <- HDIM::FOS( as.matrix(xinput), as.matrix(yinput), "cd" )
}
\keyword{models}
\keyword{regression}
