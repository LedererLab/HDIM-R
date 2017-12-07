#ifndef FOS_R_H
#define FOS_R_H

// C System-Headers
//
// C++ System headers
#include <type_traits>
#include <stdexcept>
// Eigen Headers
//
// Boost Headers
//
// Rcpp Headers
#include <Rcpp.h>
// Project Specific Headers
#include "FOS/FOS/x_fos.hpp"
#include "FOS/Solvers/SubGradientDescent/ISTA/ista.hpp"
#include "FOS/Solvers/SubGradientDescent/FISTA/fista.hpp"
#include "FOS/Solvers/CoordinateDescent/coordinate_descent.hpp"

// Functions that handle conversions between Rcpp and Eigen3 objects.

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, 1 > NumVect2Eigen( const Rcpp::NumericVector& vec ) {

    int len = vec.length();

    T* non_const_vec_data = new T[ len ];
    const double* vec_data = &vec[0];

    std::copy( vec_data, vec_data + len, non_const_vec_data );

    return Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, 1 > >( non_const_vec_data, len );
}

template < typename T >
Rcpp::NumericVector Eigen2NumVec( const Eigen::Matrix< T, Eigen::Dynamic, 1 >& vec ) {

    const T* vect_data = vec.data();
    return Rcpp::NumericVector::import( vect_data, vect_data + vec.size() );
}

template < typename T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > NumMat2Eigen( const Rcpp::NumericMatrix& mat ) {

    int rows = mat.rows();
    int cols = mat.cols();

    int eigen_mat_size = rows*cols;

    T* non_const_mat_data = new T[ eigen_mat_size ];
    const double* mat_data = &mat[0];

    std::copy( mat_data, mat_data + eigen_mat_size, non_const_mat_data );

    return Eigen::Map< Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >( non_const_mat_data, rows, cols );

}

template < typename T >
Rcpp::NumericMatrix Eigen2NumMat( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& mat ) {

    const T* mat_data = mat.data();
    return Rcpp::NumericMatrix::import( mat_data, mat_data + mat_data.size() );
}

// FOS

template < typename T >
Rcpp::List __FOS( const Rcpp::NumericMatrix& X,
                const Rcpp::NumericVector& Y,
                const std::string solver_type,
                const bool use_single_precision = false ) {

    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > mat_X = NumMat2Eigen<T>(X);
    Eigen::Matrix< T, Eigen::Dynamic, 1 > vect_Y = NumVect2Eigen<T>(Y);

    hdim::X_FOS<T> fos;

    if ( solver_type == "ista" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::ista );
    } else if ( solver_type == "screen_ista" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::screen_ista );
    } else if ( solver_type == "fista" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::fista );
    } else if ( solver_type == "screen_fista" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::screen_fista );
    } else if ( solver_type == "cd" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::cd );
    } else if ( solver_type == "screen_cd" ) {
      fos(  mat_X, vect_Y, hdim::SolverType::screen_cd );
    } else {
      fos(  mat_X, vect_Y, hdim::SolverType::screen_cd );
    }

    // Rcpp::NumericVector beta = Eigen2NumVec<double>( fos.ReturnCoefficients() );

    Rcpp::NumericVector beta = Eigen2NumVec<T>( fos.ReturnCoefficients() );

    beta.attr("names") = Rcpp::colnames(X);

    unsigned int stopping_index = fos.ReturnOptimIndex();
    double lambda = fos.ReturnLambda();
    double intercept = fos.ReturnIntercept();

    Rcpp::NumericVector support = Eigen2NumVec<int>( fos.ReturnSupport() );

    return Rcpp::List::create( Rcpp::Named("beta") = beta,
                              Rcpp::Named("index") = stopping_index,
                              Rcpp::Named("lambda") = lambda,
                              Rcpp::Named("intercept") = intercept,
                              Rcpp::Named("support") = support );

}

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

//' @title The Fast and Optimal Support Algorithm
//'
//' Description
//' @param X: An n x p design matrix.
//' @param Y: A 1 x p array representing the predictors
//' @param solver_type: The type of iterative solver used internally. Can used
//' sub-gradient descent methods or coordinate descent, both of which
//' can use GAPSAFE screening rules or not.
//' @param use_single_precision: If set to TRUE double-precision floating point values
//' will be cast to single-precision. This will result in less memory usable and faster
//' execution, but may result in numerical precision issues.
//'
//' @return A list containing the results of the regression.
//' \item{beta}{The coefficients of the regression results}
//' \item{index}{The number of the grid element where the algorithm stops, in the range 1 - 100 }
//' \item{lambda}{The regularization parameter that was deemed optimal}
//' \item{intercept}{Value of the intercept term.}
//' \item{support}{The estimated support. Enteries will be 0 if the regression coefficient was
//' zero, and 1 if it was non-zero and signifigant.}
//'
//' @description Perform an L1 regularized linear regression using the Fast and Optimal Support
//'  algorithm.
//'
//' @references Néhémy Lim, Johannes Lederer (2016)
//'   \emph{Efficient Feature Selection With Large and High-dimensional Data},
//'      \url{https://arxiv.org/abs/1609.07195}\cr
//'   \emph{Pre-print via ArXiv}\cr
//'   \url{https://arxiv.org}\cr
//'
//' @author Benjamin J Phillips e-mail:bejphil@uw.edu
//'
//' @examples
//' library(HDIM)
//'
//' dataset <- matrix(rexp(200, rate=.1), ncol=20)
//'
//' yinput <- dataset[, 1, drop = FALSE]
//' xinput <- dataset[, names(dataset) != names(yinput)]
//'
//' fos_fit <- HDIM::FOS( as.matrix(xinput), as.matrix(yinput), "cd" )
// [[Rcpp::export]]
Rcpp::List FOS( const Rcpp::NumericMatrix& X,
                const Rcpp::NumericVector& Y,
                const std::string solver_type,
                const bool use_single_precision = false ) {

    if( use_single_precision == true ) {
      return __FOS<float>( X, Y, solver_type );
    } else {
      return __FOS<double>( X, Y, solver_type );
    }

}

// Coordinate Descent

struct fallthrough {};

template< typename T, typename CC >
Eigen::Matrix< T, Eigen::Dynamic, 1 > __CD_helper( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                                                   const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                                                   const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
                                                   const T Lambda,
                                                   const CC convergence_criteria,
                                                   const bool use_screening_rules ) {

  if( use_screening_rules ) {
    hdim::LazyCoordinateDescent<T,hdim::internal::ScreeningSolver<T>> cd_solver( X, Y, Beta_0 );
    return cd_solver( X, Y, Beta_0, Lambda, convergence_criteria );
  } else {
    hdim::LazyCoordinateDescent<T,hdim::internal::Solver<T>> cd_solver( X, Y, Beta_0 );
    return cd_solver( X, Y, Beta_0, Lambda, convergence_criteria );
  }

}

template <typename T>
Rcpp::NumericVector __CD( const Rcpp::NumericMatrix X,
                          const Rcpp::NumericVector Y,
                          const Rcpp::NumericVector Beta_0,
                          const double Lambda,
                          const T convergence_criteria,
                          const bool use_screening_rules = false,
                          const bool use_single_precision = false ) {

    std::cout << "Using default!" << std::endl;
    return Rcpp::NumericVector();

}

template <>
Rcpp::NumericVector __CD<double>( const Rcpp::NumericMatrix X,
                                  const Rcpp::NumericVector Y,
                                  const Rcpp::NumericVector Beta_0,
                                  const double Lambda,
                                  const double convergence_criteria,
                                  const bool use_screening_rules,
                                  const bool use_single_precision ) {

  if( use_single_precision == false ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    return Eigen2NumVec<double>( __CD_helper<double,double>( mat_X,
                                                             vect_Y,
                                                             vect_Beta_0,
                                                             Lambda,
                                                             convergence_criteria,
                                                             use_screening_rules ) );

  } else {

    Eigen::MatrixXf mat_X = NumMat2Eigen<float>(X);
    Eigen::VectorXf vect_Y = NumVect2Eigen<float>(Y);
    Eigen::VectorXf vect_Beta_0 = NumVect2Eigen<float>(Beta_0);

    return Eigen2NumVec<float>( __CD_helper<float,float>( mat_X,
                                                          vect_Y,
                                                          vect_Beta_0,
                                                          Lambda,
                                                          convergence_criteria,
                                                          use_screening_rules ) );

  }

}

template <>
Rcpp::NumericVector __CD<unsigned int>( const Rcpp::NumericMatrix X,
                                        const Rcpp::NumericVector Y,
                                        const Rcpp::NumericVector Beta_0,
                                        const double Lambda,
                                        const unsigned int convergence_criteria,
                                        const bool use_screening_rules,
                                        const bool use_single_precision ) {

  if( use_single_precision == false ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    return Eigen2NumVec<double>( __CD_helper<double,unsigned int>( mat_X,
                                                                   vect_Y,
                                                                   vect_Beta_0,
                                                                   Lambda,
                                                                   convergence_criteria,
                                                                   use_screening_rules ) );

  } else {

    Eigen::MatrixXf mat_X = NumMat2Eigen<float>(X);
    Eigen::VectorXf vect_Y = NumVect2Eigen<float>(Y);
    Eigen::VectorXf vect_Beta_0 = NumVect2Eigen<float>(Beta_0);

    return Eigen2NumVec<float>( __CD_helper<float,unsigned int>( mat_X,
                                                                 vect_Y,
                                                                 vect_Beta_0,
                                                                 Lambda,
                                                                 convergence_criteria,
                                                                 use_screening_rules ) );

  }

}

// [[Rcpp::export]]
Rcpp::NumericVector CoordinateDescent( Rcpp::NumericMatrix& X,
                                       Rcpp::NumericVector& Y,
                                       Rcpp::NumericVector& Beta_0,
                                       const double Lambda,
                                       SEXP convergence_criteria,
                                       const bool use_screening_rules = false,
                                       const bool use_single_precision = false ) {

    switch (TYPEOF(convergence_criteria)) {
        case INTSXP: {
            return __CD<unsigned int>( X,
                                       Y,
                                       Beta_0,
                                       Lambda,
                                       static_cast<unsigned int>(INTEGER(convergence_criteria)[0]),
                                       use_screening_rules,
                                       use_single_precision );
        }
        case REALSXP: {
            return __CD<double>( X,
                                 Y,
                                 Beta_0,
                                 Lambda,
                                 static_cast<double>(REAL(convergence_criteria)[0]),
                                 use_screening_rules,
                                 use_single_precision );
        }
        default: {
            Rcpp::warning("Unmatched SEXPTYPE!");
            throw std::invalid_argument( "Coordinate Descent can't be used for this type!" );
            return __CD<fallthrough>( X,
                                      Y,
                                      Beta_0,
                                      Lambda,
                                      fallthrough(),
                                      use_screening_rules,
                                      use_single_precision );
        }
    }

}

// ISTA

template< typename T, typename CC >
Eigen::Matrix< T, Eigen::Dynamic, 1 > __ISTA_helper( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                                                     const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                                                     const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
                                                     const T Lambda,
                                                     const T L_0,
                                                     const CC convergence_criteria,
                                                     const bool use_screening_rules ) {

  if( use_screening_rules ) {
    hdim::ISTA<T,hdim::internal::ScreeningSolver<T>> ista_solver( L_0 );
    return ista_solver( X, Y, Beta_0, Lambda, convergence_criteria );
  } else {
    hdim::ISTA<T,hdim::internal::Solver<T>> ista_solver( L_0 );
    return ista_solver( X, Y, Beta_0, Lambda, convergence_criteria );
  }

}

template <typename T>
Rcpp::NumericVector __ISTA( const Rcpp::NumericMatrix X,
                            const Rcpp::NumericVector Y,
                            const Rcpp::NumericVector Beta_0,
                            const double Lambda,
                            const T convergence_criteria,
                            const double L_0 = 0.1,
                            const bool use_screening_rules = false,
                            const bool use_single_precision = false ) {

    std::cout << "Using default!" << std::endl;
    return Rcpp::NumericVector();

}

template <>
Rcpp::NumericVector __ISTA<double>( const Rcpp::NumericMatrix X,
                                    const Rcpp::NumericVector Y,
                                    const Rcpp::NumericVector Beta_0,
                                    const double Lambda,
                                    const double convergence_criteria,
                                    const double L_0,
                                    const bool use_screening_rules,
                                    const bool use_single_precision ) {

  if( use_single_precision == false ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    return Eigen2NumVec<double>( __ISTA_helper<double,double>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  } else {

    Eigen::MatrixXf mat_X = NumMat2Eigen<float>(X);
    Eigen::VectorXf vect_Y = NumVect2Eigen<float>(Y);
    Eigen::VectorXf vect_Beta_0 = NumVect2Eigen<float>(Beta_0);

    return Eigen2NumVec<float>( __ISTA_helper<float,float>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  }

}

template <>
Rcpp::NumericVector __ISTA<unsigned int>( const Rcpp::NumericMatrix X,
                                          const Rcpp::NumericVector Y,
                                          const Rcpp::NumericVector Beta_0,
                                          const double Lambda,
                                          const unsigned int convergence_criteria,
                                          const double L_0,
                                          const bool use_screening_rules,
                                          const bool use_single_precision ) {

  if( use_single_precision == false ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    return Eigen2NumVec<double>( __ISTA_helper<double,unsigned int>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  } else {

    Eigen::MatrixXf mat_X = NumMat2Eigen<float>(X);
    Eigen::VectorXf vect_Y = NumVect2Eigen<float>(Y);
    Eigen::VectorXf vect_Beta_0 = NumVect2Eigen<float>(Beta_0);

    return Eigen2NumVec<float>( __ISTA_helper<float,unsigned int>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  }

}

// [[Rcpp::export]]
Rcpp::NumericVector ISTA( const Rcpp::NumericMatrix X,
                                       const Rcpp::NumericVector Y,
                                       const Rcpp::NumericVector Beta_0,
                                       const double Lambda,
                                       const SEXP convergence_criteria,
                                       const double L_0 = 0.1,
                                       const bool use_screening_rules = false,
                                       const bool use_single_precision = false ) {

    switch (TYPEOF(convergence_criteria)) {
        case INTSXP: {
            return __ISTA<unsigned int>( X,
                                         Y,
                                         Beta_0,
                                         Lambda,
                                         static_cast<unsigned int>(INTEGER(convergence_criteria)[0]),
                                         L_0,
                                         use_screening_rules,
                                         use_single_precision );
        }
        case REALSXP: {
            return __ISTA<double>( X,
                                   Y,
                                   Beta_0,
                                   Lambda,
                                   static_cast<double>(REAL(convergence_criteria)[0]),
                                   L_0,
                                   use_screening_rules,
                                   use_single_precision );
        }
        default: {
            Rcpp::warning("Unmatched SEXPTYPE!");
            throw std::invalid_argument( "Coordinate Descent can't be used for this type!" );
            return __ISTA<fallthrough>( X,
                                        Y,
                                        Beta_0,
                                        Lambda,
                                        fallthrough(),
                                        L_0,
                                        use_screening_rules,
                                        use_single_precision );
        }
    }

}

// FISTA

template <typename T>
Rcpp::NumericVector __FISTA( const Rcpp::NumericMatrix X,
                            const Rcpp::NumericVector Y,
                            const Rcpp::NumericVector Beta_0,
                            const double Lambda,
                            const T convergence_criteria,
                            const double L_0 = 0.1,
                            const bool use_screening_rules = false,
                            const bool use_single_precision = false ) {

    std::cout << "Using default!" << std::endl;
    return Rcpp::NumericVector();

}

template< typename T, typename CC >
Eigen::Matrix< T, Eigen::Dynamic, 1 > __FISTA_helper( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                                                     const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                                                     const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
                                                     const T Lambda,
                                                     const T L_0,
                                                     const CC convergence_criteria,
                                                     const bool use_screening_rules ) {

  if( use_screening_rules ) {
    hdim::FISTA<T,hdim::internal::ScreeningSolver<T>> fista_solver( Beta_0, L_0 );
    return fista_solver( X, Y, Beta_0, Lambda, convergence_criteria );
  } else {
    hdim::FISTA<T,hdim::internal::Solver<T>> fista_solver( Beta_0, L_0 );
    return fista_solver( X, Y, Beta_0, Lambda, convergence_criteria );
  }

}

template <>
Rcpp::NumericVector __FISTA<double>( const Rcpp::NumericMatrix X,
                                    const Rcpp::NumericVector Y,
                                    const Rcpp::NumericVector Beta_0,
                                    const double Lambda,
                                    const double convergence_criteria,
                                    const double L_0,
                                    const bool use_screening_rules,
                                    const bool use_single_precision ) {

  if( use_single_precision == false ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    return Eigen2NumVec<double>( __FISTA_helper<double,double>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  } else {

    Eigen::MatrixXf mat_X = NumMat2Eigen<float>(X);
    Eigen::VectorXf vect_Y = NumVect2Eigen<float>(Y);
    Eigen::VectorXf vect_Beta_0 = NumVect2Eigen<float>(Beta_0);

    return Eigen2NumVec<float>( __FISTA_helper<float,float>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  }

}

template <>
Rcpp::NumericVector __FISTA<unsigned int>( const Rcpp::NumericMatrix X,
                                          const Rcpp::NumericVector Y,
                                          const Rcpp::NumericVector Beta_0,
                                          const double Lambda,
                                          const unsigned int convergence_criteria,
                                          const double L_0,
                                          const bool use_screening_rules,
                                          const bool use_single_precision ) {

  if( use_single_precision == false ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    return Eigen2NumVec<double>( __FISTA_helper<double,unsigned int>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  } else {

    Eigen::MatrixXf mat_X = NumMat2Eigen<float>(X);
    Eigen::VectorXf vect_Y = NumVect2Eigen<float>(Y);
    Eigen::VectorXf vect_Beta_0 = NumVect2Eigen<float>(Beta_0);

    return Eigen2NumVec<float>( __FISTA_helper<float,unsigned int>( mat_X, vect_Y, vect_Beta_0, Lambda, convergence_criteria, L_0, use_screening_rules ) );

  }

}

// [[Rcpp::export]]
Rcpp::NumericVector FISTA( const Rcpp::NumericMatrix X,
                                       const Rcpp::NumericVector Y,
                                       const Rcpp::NumericVector Beta_0,
                                       const double Lambda,
                                       const SEXP convergence_criteria,
                                       const double L_0 = 0.1,
                                       const bool use_screening_rules = false,
                                       const bool use_single_precision = false ) {

    switch (TYPEOF(convergence_criteria)) {
        case INTSXP: {
            return __FISTA<unsigned int>( X,
                                         Y,
                                         Beta_0,
                                         Lambda,
                                         static_cast<unsigned int>(INTEGER(convergence_criteria)[0]),
                                         L_0,
                                         use_screening_rules,
                                         use_single_precision );
        }
        case REALSXP: {
            return __FISTA<double>( X,
                                   Y,
                                   Beta_0,
                                   Lambda,
                                   static_cast<double>(REAL(convergence_criteria)[0]),
                                   L_0,
                                   use_screening_rules,
                                   use_single_precision );
        }
        default: {
            Rcpp::warning("Unmatched SEXPTYPE!");
            throw std::invalid_argument( "Coordinate Descent can't be used for this type!" );
            return __FISTA<fallthrough>( X,
                                      Y,
                                      Beta_0,
                                      Lambda,
                                      fallthrough(),
                                      L_0,
                                      use_screening_rules,
                                      use_single_precision );
        }
    }

}

#endif // FOS_R_H
