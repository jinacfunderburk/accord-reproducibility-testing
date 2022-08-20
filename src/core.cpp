#include <iostream>
// #include <stdlib.h>
// #include <stdio.h>
// #include <math.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>

using namespace Eigen;
using namespace std;

double sgn(double val) {
    return (double(0) < val) - (val < double(0));
}

double sthresh(double x, double t ){
    return sgn(x) * max(abs(x)-t, 0.0);
}

void sthreshmat(MatrixXd & x,
                double tau,
                const MatrixXd & t){

    MatrixXd tmp1(x.cols(), x.cols());
    MatrixXd tmp2(x.cols(), x.cols());

    tmp1 = x.array().unaryExpr(ptr_fun(sgn));
    tmp2 = (x.cwiseAbs() - tau*t).cwiseMax(0.0);

    x = tmp1.cwiseProduct(tmp2);

    return;
}

inline double shrink(double a, double b) {
    if (b < fabs(a)) {
        if (a > 0) return(a-b);
        else       return(a+b);
    } else {
        return(0.0);
    }
}

//SparseMatrix<double> ccista(
//    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat
//    // double lambda2,
//    // double epstol,
//    // int maxitr,
//    // int steptype
//    ) 
//{   
//    int p = LambdaMat.cols();
//    return LambdaMat.sparseView();
//}
 
void grad_h1(Ref<SparseMatrix<double>> X_, Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S_, Ref<MatrixXd> G_) {
    MatrixXd W = S_ * X_;
    // G = -
    std::cout << W << std::endl;
}

SparseMatrix<double> ccista(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S, 
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double epstol,
    int maxitr
    ) {

    int p = S.cols();

    SparseMatrix<double> X(p, p);
    X.setIdentity();
    SparseMatrix<double> Xn(p, p);
    MatrixXd G(p, p);               // gradient
    MatrixXd W(p, p);               // S*X

    // double tau_ = 1.0;
    // double c_ = 0.5;
    // int inner_itr_count = 0;        // linesearch iteration count

    grad_h1(X, S, G);

    return X;
}
    // while (true) {

    //     // reset step size and search multipler to defaults
    //     tau = tau_;
    //     c = c_;

    //     G = grad_h1(X, S);

    //     inner_itr_count = 0;
    //     while (true) {

    //         step = X - tau*G;

    //         break;

    //     }

    //     break;

    // }
 
    // DiagonalMatrix<double, Dynamic> XdiagM(p);
    // SparseMatrix<double> Step;
    // MatrixXd W = S * X;
    // MatrixXd Wn(p, p);
    // MatrixXd Gn(p, p);
    // MatrixXd subg(p, p);
    // MatrixXd tmp(p, p);
 
    // double h = - X.diagonal().array().log().sum() + 0.5 * ((X*W).diagonal().sum()) + (lambda2 * pow(X.norm(), 2));
    // double hn = 0;
    // double Qn = 0;
    // double subgnorm, Xnnorm;
    // double tau;
    // double taun = 1.0;
    // double c = 0.5;
    // int itr = 0;
    // int loop = 1;
    // int diagitr = 0;
    // int backitr = 0;
 
    // XdiagM.diagonal() = - X.diagonal();
    // G = XdiagM.inverse();
    // G += 0.5 * (W + W.transpose());
    // if (lambda2 > 0) { G += lambda2 * 2.0 * X; }
 
    // while (loop != 0){
 
    //     tau = taun;
    //     diagitr = 0;
    //     backitr = 0;
 
    //     while ( 1  && (backitr < 100)) { // back-tracking line search
 
    //         if (diagitr != 0 || backitr != 0) { tau = tau * c; }
 
    //         tmp = MatrixXd(X) - tau*G;
    //         sthreshmat(tmp, tau, LambdaMat);
    //         Xn = tmp.sparseView();
 
    //         if (Xn.diagonal().minCoeff() < 1e-8 && diagitr < 10) { diagitr += 1; continue; }
 
    //         Step = Xn - X;
    //         Wn = S * Xn;
    //         Qn = h + Step.cwiseProduct(G).sum() + (1/(2*tau))*pow(Step.norm(),2);
    //         hn = - Xn.diagonal().cwiseAbs().array().log().sum() + 0.5 * (Xn.cwiseProduct(Wn).sum());
 
    //         if (lambda2 > 0) { hn += lambda2 * pow(Xn.norm(), 2); }
 
    //         if (hn > Qn) { backitr += 1; } else { break; }
 
    //     }
 
    //     XdiagM.diagonal() = - Xn.diagonal();
    //     Gn = XdiagM.inverse();
    //     Gn += 0.5 * (Wn + Wn.transpose());
 
    //     if (lambda2 > 0) { Gn += lambda2 * 2 * MatrixXd(Xn); }
 
    //     if ( steptype == 0 ) {
    //         taun = ( Step * Step ).eval().diagonal().array().sum() / (Step.cwiseProduct( Gn - G ).sum());
    //     } else if ( steptype == 1 ) {
    //         taun = 1;
    //     } else if ( steptype == 2 ){
    //       taun = tau;
    //     }
 
    //     tmp = MatrixXd(Xn).array().unaryExpr(ptr_fun(sgn));
    //     tmp = Gn + tmp.cwiseProduct(LambdaMat);
    //     subg = Gn;
    //     sthreshmat(subg, 1.0, LambdaMat);
    //     subg = (MatrixXd(Xn).array() != 0).select(tmp, subg);
    //     subgnorm = subg.norm();
    //     Xnnorm = Xn.norm();
    
    //     X = Xn;
    //     h = hn;
    //     G = Gn;
 
    //     itr += 1;
 
    //     loop = int((itr < maxitr) && (subgnorm/Xnnorm > epstol));
 
    // }
 
    // return X;
    // return SparseView<MatrixXf>(S); 
    // return S;
// }
//  // ista algorithm
//  void ccista(MatrixXd&       S,
//              SparseMatrix<double>& X,
//              MatrixXd&       LambdaMat,
//              double  lambda2,
//              double  epstol,
//              int     maxitr,
//              int     steptype
//                )
//  {
//  
//    int p = S.cols();
//    DiagonalMatrix<double, Dynamic> XdiagM(p);
//    SparseMatrix<double> Xn;
//    SparseMatrix<double> Step;
//    MatrixXd W = S * X;
//    MatrixXd Wn(p, p);
//    MatrixXd G(p, p);
//    MatrixXd Gn(p, p);
//    MatrixXd subg(p, p);
//    MatrixXd tmp(p, p);
//  
//    double h = - X.diagonal().array().log().sum() + 0.5 * ((X*W).diagonal().sum()) + (lambda2 * pow(X.norm(), 2));
//    double hn = 0;
//    double Qn = 0;
//    double subgnorm, Xnnorm;
//    double tau;
//    double taun = 1.0;
//    double c = 0.5;
//    int itr = 0;
//    int loop = 1;
//    int diagitr = 0;
//    int backitr = 0;
//  
//    XdiagM.diagonal() = - X.diagonal();
//    G = XdiagM.inverse();
//    G += 0.5 * (W + W.transpose());
//    if (lambda2 > 0) { G += lambda2 * 2.0 * X; }
//  
//    while (loop != 0){
//  
//      tau = taun;
//      diagitr = 0;
//      backitr = 0;
//  
//      while ( 1  && (backitr < 100)) { // back-tracking line search
//  
//        if (diagitr != 0 || backitr != 0) { tau = tau * c; }
//  
//        tmp = MatrixXd(X) - tau*G;
//        sthreshmat(tmp, tau, LambdaMat);
//        Xn = tmp.sparseView();
//  
//        if (Xn.diagonal().minCoeff() < 1e-8 && diagitr < 10) { diagitr += 1; continue; }
//  
//        Step = Xn - X;
//        Wn = S * Xn;
//        Qn = h + Step.cwiseProduct(G).sum() + (1/(2*tau))*pow(Step.norm(),2);
//        hn = - Xn.diagonal().cwiseAbs().array().log().sum() + 0.5 * (Xn.cwiseProduct(Wn).sum());
//  
//        if (lambda2 > 0) { hn += lambda2 * pow(Xn.norm(), 2); }
//  
//        if (hn > Qn) { backitr += 1; } else { break; }
//  
//      }
//  
//      XdiagM.diagonal() = - Xn.diagonal();
//      Gn = XdiagM.inverse();
//      Gn += 0.5 * (Wn + Wn.transpose());
//  
//      if (lambda2 > 0) { Gn += lambda2 * 2 * MatrixXd(Xn); }
//  
//      if ( steptype == 0 ) {
//        taun = ( Step * Step ).eval().diagonal().array().sum() / (Step.cwiseProduct( Gn - G ).sum());
//      } else if ( steptype == 1 ) {
//        taun = 1;
//      } else if ( steptype == 2 ){
//        taun = tau;
//      }
//  
//      tmp = MatrixXd(Xn).array().unaryExpr(ptr_fun(sgn));
//      tmp = Gn + tmp.cwiseProduct(LambdaMat);
//      subg = Gn;
//      sthreshmat(subg, tau, LambdaMat);
//      subg = (MatrixXd(Xn).array() != 0).select(tmp, subg);
//      subgnorm = subg.norm();
//      Xnnorm = Xn.norm();
//      
//      X = Xn;
//      h = hn;
//      G = Gn;
//  
//      itr += 1;
//  
//      loop = int((itr < maxitr) && (subgnorm/Xnnorm > epstol));
//  
//    }
// 
// }




// int add(int i, int j) {
//     return i + j;
// }
// 
// // matrix determinant
// double det(const Eigen::MatrixXd &xs) {
//   return xs.determinant();
// }
// 
// // matrix inverse
// Eigen::MatrixXd inv(const Eigen::MatrixXd &xs) {
//   return xs.inverse();
// }
// 
// // modify input matrix in place
// void mutate(Eigen::Ref<Eigen::MatrixXd> x) {
//     x(0, 0) = 30.201;
// }
// 
// // modify input vector in place and return a reference to it
// Eigen::Ref<Eigen::VectorXd> incr_vector(Eigen::Ref<Eigen::VectorXd> m, double v) {
//     m += Eigen::VectorXd::Constant(m.rows(), v);
//     return m;
// }
// 
// // modify input matrix in place and return a reference to it
// Eigen::Ref<Eigen::MatrixXd> incr_matrix(Eigen::Ref<Eigen::MatrixXd> m, double v) {
//     m += Eigen::MatrixXd::Constant(m.rows(), m.cols(), v);
//     return m;
// }
// 
// // modify input matrix in place and return a reference to it
// // see https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html?highlight=eigendref#storage-orders
// // py::EigenDRef<Eigen::MatrixXd> incr_matrix_any(py::EigenDRef<Eigen::MatrixXd> m, double v); // short version
// Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> incr_matrix_any(Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> m, double v) {
//     m += Eigen::MatrixXd::Constant(m.rows(), m.cols(), v);
//     return m;
// }
// 
// Eigen::SparseMatrix<float> sparsify_c(Eigen::MatrixXf &xs) { 
//     return Eigen::SparseView<Eigen::MatrixXf>(xs); 
// }
// 
// Eigen::SparseMatrix<float, Eigen::RowMajor> sparsify_r(Eigen::MatrixXf &xs) { 
//     return Eigen::SparseView<Eigen::MatrixXf>(xs); 
// }