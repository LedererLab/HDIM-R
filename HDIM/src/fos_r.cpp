#ifndef FOS_R_H
#define FOS_R_H

// C System-Headers
//
// C++ System headers
#include <type_traits>
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

template < typename T >
Rcpp::List FOSBase( const Rcpp::NumericMatrix& X,
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
      return FOSBase<float>( X, Y, solver_type );
    } else {
      return FOSBase<double>( X, Y, solver_type );
    }

}

// [[Rcpp::export]]
Rcpp::NumericVector CD( Rcpp::NumericMatrix X,
                        Rcpp::NumericVector Y,
                        Rcpp::NumericVector Beta_0,
                        double lambda,
                        unsigned int num_iterations ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::LazyCoordinateDescent<double,hdim::internal::Solver<double>> cd_solver( mat_X, vect_Y, vect_Beta_0 );

    return Eigen2NumVec<double>( cd_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector CDSR( Rcpp::NumericMatrix X,
                        Rcpp::NumericVector Y,
                        Rcpp::NumericVector Beta_0,
                        double lambda,
                        unsigned int num_iterations ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::LazyCoordinateDescent<double,hdim::internal::ScreeningSolver<double>> cd_solver( mat_X, vect_Y, vect_Beta_0 );

    return Eigen2NumVec<double>( cd_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector CDDG( Rcpp::NumericMatrix X,
                        Rcpp::NumericVector Y,
                        Rcpp::NumericVector Beta_0,
                        double lambda,
                        double duality_gap_target ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::LazyCoordinateDescent<double,hdim::internal::Solver<double>> cd_solver( mat_X, vect_Y, vect_Beta_0 );

    return Eigen2NumVec<double>( cd_solver( mat_X, vect_Y, vect_Beta_0, lambda, duality_gap_target ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector CDSRDG( Rcpp::NumericMatrix X,
	                Rcpp::NumericVector Y,
	                Rcpp::NumericVector Beta_0,
	                double lambda,
	                double duality_gap_target ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::LazyCoordinateDescent<double,hdim::internal::ScreeningSolver<double>> cd_solver( mat_X, vect_Y, vect_Beta_0 );

    return Eigen2NumVec<double>( cd_solver( mat_X, vect_Y, vect_Beta_0, lambda, duality_gap_target ) );
}


// [[Rcpp::export]]
Rcpp::NumericVector ISTA( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          unsigned int num_iterations,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::ISTA<double,hdim::internal::Solver<double>> ista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( ista_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector ISTASR( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          unsigned int num_iterations,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::ISTA<double,hdim::internal::ScreeningSolver<double>> ista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( ista_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector ISTA_DG( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          double duality_gap_target,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::ISTA<double,hdim::internal::Solver<double>> ista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( ista_solver( mat_X, vect_Y, vect_Beta_0, lambda, duality_gap_target ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector ISTASR_DG( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          double duality_gap_target,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::ISTA<double,hdim::internal::ScreeningSolver<double>> ista_solver( static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( ista_solver( mat_X, vect_Y, vect_Beta_0, lambda, duality_gap_target ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector FISTA( Rcpp::NumericMatrix X,
                           Rcpp::NumericVector Y,
                           Rcpp::NumericVector Beta_0,
                           double lambda,
                           unsigned int num_iterations,
                           double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::FISTA<double,hdim::internal::Solver<double>> fista_solver( vect_Beta_0, static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( fista_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector FISTASR( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          unsigned int num_iterations,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::FISTA<double,hdim::internal::ScreeningSolver<double>> fista_solver( vect_Beta_0, static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( fista_solver( mat_X, vect_Y, vect_Beta_0, lambda, num_iterations ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector FISTA_DG( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          double duality_gap_target,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::FISTA<double,hdim::internal::Solver<double>> fista_solver( vect_Beta_0, static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( fista_solver( mat_X, vect_Y, vect_Beta_0, lambda, duality_gap_target ) );
}

// [[Rcpp::export]]
Rcpp::NumericVector FISTASR_DG( Rcpp::NumericMatrix X,
                          Rcpp::NumericVector Y,
                          Rcpp::NumericVector Beta_0,
                          double lambda,
                          double duality_gap_target,
                          double L_0 = 0.1 ) {

    Eigen::MatrixXd mat_X = NumMat2Eigen<double>(X);
    Eigen::VectorXd vect_Y = NumVect2Eigen<double>(Y);
    Eigen::VectorXd vect_Beta_0 = NumVect2Eigen<double>(Beta_0);

    hdim::FISTA<double,hdim::internal::ScreeningSolver<double>> fista_solver( vect_Beta_0, static_cast<double>( L_0 ) );

    return Eigen2NumVec<double>( fista_solver( mat_X, vect_Y, vect_Beta_0, lambda, duality_gap_target ) );
}

#endif // FOS_R_H
