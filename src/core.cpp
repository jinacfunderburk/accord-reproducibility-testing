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

SparseMatrix<double> cceista(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S, 
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double epstol,
    int maxitr,
    Ref<VectorXi> hist_inner_itr_count,
    Ref<VectorXd> hist_norm_diff,
    Ref<VectorXd> hist_hn
    ) {

    int p = S.cols();

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // S*X

    MatrixXd subg(p, p);
    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd subgrad_h2(p, p);              // subgradient of h2
    MatrixXd tmp(p, p);

    double Q, hn, h1, h1n, norm_diff;
    double tau, tau_ = 1.0;
    double c_ = 0.5;

    X.setIdentity();                        // initial guess: X = I
    W = X * S;
    grad_h1 = -1*X.diagonal().asDiagonal().inverse();
    grad_h1 += W;
    h1 = - X.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(X.transpose())*W).trace();

    int outer_itr_count, inner_itr_count;   // iteration counts

    outer_itr_count = 0;
    while (true) {
 
        tau = tau_;

        inner_itr_count = 0;
        while (true) {

            tmp = MatrixXd(X) - tau*grad_h1;
            sthreshmat(tmp, tau, LambdaMat);
            Xn = tmp.sparseView();

            if (tmp.diagonal().minCoeff() > 0) {

                Step = Xn - X;
                Wn = Xn * S;

                h1n = - Xn.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(Xn.transpose())*Wn).trace();
                Q = h1 + Step.cwiseProduct(grad_h1).sum() + (0.5/tau)*Step.squaredNorm();

                if (h1n <= Q)
                    break; 

            }

            // getting here means:
            // 1. at least one diagonal element is negative
            // 2. sufficient descent condition wasn't satisfied
            tau *= c_;
            inner_itr_count += 1;

        }

        grad_h1 = -1*Xn.diagonal().asDiagonal().inverse();
        grad_h1 += Wn;

        // subgrad_h2(X, lambda1, G)
        // subgradient when X_ij is zero. set as close to zero as possible
        subgrad_h2 = -1 * LambdaMat.array() * grad_h1.array().sign() * (grad_h1.array().abs()/LambdaMat.array()).min(1);
        // subgradient when X_ij is not zero. set lambda1 * sign(X)
        for (int k=0; k<X.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(X, k); it; ++it) {
                subgrad_h2(it.row(), it.col()) = LambdaMat(it.row(), it.col()) * sgn(it.value());
            }
        }

        subg = grad_h1 + subgrad_h2;
        norm_diff = Step.norm();
        hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();

        hist_inner_itr_count(outer_itr_count) = inner_itr_count;
        hist_norm_diff(outer_itr_count) = norm_diff;
        hist_hn(outer_itr_count) = hn;

        outer_itr_count += 1;

        if (norm_diff < epstol || outer_itr_count >= maxitr) {
            if (outer_itr_count <= maxitr) {
                hist_inner_itr_count(outer_itr_count) = -1;
                hist_norm_diff(outer_itr_count) = -1;
                hist_hn(outer_itr_count) = -1;
            }
            break;
        } else {
            h1 = h1n;
            X = Xn;
            W = Wn;
        }
    }

    Xn = 0.5*(SparseMatrix<double>(Xn.diagonal().asDiagonal()) * Xn) + 0.5*(SparseMatrix<double>(Xn.transpose()) * Xn.diagonal().asDiagonal());

    return Xn;
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