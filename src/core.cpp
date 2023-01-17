#include <iostream>
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

SparseMatrix<double> cce(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S, 
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diagonal,
    Ref<VectorXd> hist_norm_diff,
    Ref<VectorXd> hist_hn
    ) {

    int p = S.cols();

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // X*S

    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd tmp(p, p);
    ArrayXd y(p);                           // diagonal elements

    double hn, norm_diff;

    X.setIdentity();                        // initial guess: X = I
    grad_h1 = X * S;

    int itr_count;                          // iteration counts

    itr_count = 0;
    while (true) {

        tmp = MatrixXd(X) - tau*grad_h1;

        if (penalize_diagonal == true) {
            y = tmp.diagonal().array() - tau*LambdaMat.diagonal().array();
            y = 0.5 * (y+(y.pow(2.0)+4*tau*Eigen::VectorXd::Ones(p).array()).sqrt());
            sthreshmat(tmp, tau, LambdaMat);
            tmp.diagonal() = y;
            Xn = tmp.sparseView();
        } else {
            y = tmp.diagonal().array();
            y = 0.5 * (y+(y.pow(2.0)+4*tau*Eigen::VectorXd::Ones(p).array()).sqrt());
            sthreshmat(tmp, tau, LambdaMat);
            tmp.diagonal() = y;
            Xn = tmp.sparseView();
        }
       
        Wn = Xn * S;
        grad_h1 = Wn;

        norm_diff = (Xn - X).norm();
        hn = - Xn.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(Xn.transpose())*Wn).trace() + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();

        hist_norm_diff(itr_count) = norm_diff;
        hist_hn(itr_count) = hn;

        itr_count += 1;

        if (norm_diff < epstol || itr_count >= maxitr) {
            if (itr_count <= maxitr) {
                hist_norm_diff(itr_count) = -1;
                hist_hn(itr_count) = -1;
            }
            break;
        } else {
            X = Xn;
            W = Wn;
        }
    }

    Xn = 0.5*(SparseMatrix<double>(Xn.diagonal().asDiagonal()) * Xn) + 0.5*(SparseMatrix<double>(Xn.transpose()) * Xn.diagonal().asDiagonal());

    return Xn;
}