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
 
SparseMatrix<double> ccista(
// void ccista(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S, 
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double epstol,
    int maxitr,
    Ref<VectorXi> hist_inner_itr_count,
    Ref<VectorXd> hist_delta_subg,
    Ref<VectorXd> hist_hn
    ) {

    int p = S.cols();

    // double epstol=1e-5;
    // int maxitr=100;
    // VectorXi hist_inner_itr_count = VectorXi::Zero(maxitr);
    // VectorXd hist_delta_subg = VectorXd::Zero(maxitr);
    // VectorXd hist_hn = VectorXd::Zero(maxitr);

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // S*X

    MatrixXd subg(p, p);
    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd subgrad_h2(p, p);              // subgradient of h2
    MatrixXd tmp(p, p);

    double Q, hn, h1, h1n, delta_subg;
    double tau, tau_ = 1.0;
    double c_ = 0.5;

    X.setIdentity();            // initial guess: X = I
    W = S * X;
    grad_h1 = -1*X.diagonal().asDiagonal().inverse();
    grad_h1 += 0.5 * (W + W.transpose());
    h1 = - X.diagonal().array().log().sum() + 0.5 * (X.cwiseProduct(W).sum());

    int outer_itr_count, inner_itr_count;        // iteration counts

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
                Wn = S * Xn;

                h1n = - Xn.diagonal().array().log().sum() + 0.5 * (Xn.cwiseProduct(Wn).sum());
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
        grad_h1 += 0.5 * (Wn + Wn.transpose());

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
        delta_subg = subg.norm()/Xn.norm();
        hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();

        hist_inner_itr_count(outer_itr_count) = inner_itr_count;
        hist_delta_subg(outer_itr_count) = delta_subg;
        hist_hn(outer_itr_count) = hn;

        if (delta_subg < epstol || outer_itr_count > maxitr) {
            
            if (outer_itr_count < maxitr) {
                hist_inner_itr_count(outer_itr_count+1) = -1;
                hist_delta_subg(outer_itr_count+1) = -1;
                hist_hn(outer_itr_count+1) = -1;
            }

            break;
        } else {
            h1 = h1n;
            X = Xn;
            W = Wn;
        }

        outer_itr_count += 1;

    }
    return Xn;

    // return X;
}

SparseMatrix<double> cceista(
// void ccista(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S, 
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> Omega_star,
    double epstol,
    int maxitr,
    Ref<VectorXi> hist_inner_itr_count,
    Ref<VectorXd> hist_delta_updates,
    Ref<VectorXd> hist_hn,
    Ref<VectorXd> hist_norm,
    Ref<VectorXd> hist_iter_time
    ) {

    int p = S.cols();

    // double epstol=1e-5;
    // int maxitr=100;
    // VectorXi hist_inner_itr_count = VectorXi::Zero(maxitr);
    // VectorXd hist_delta_subg = VectorXd::Zero(maxitr);
    // VectorXd hist_hn = VectorXd::Zero(maxitr);

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // S*X

    MatrixXd subg(p, p);
    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd subgrad_h2(p, p);              // subgradient of h2
    MatrixXd tmp(p, p);

    double Q, hn, h1, h1n, delta_updates;
    double tau, tau_ = 1.0;
    double c_ = 0.5;

    X.setIdentity();            // initial guess: X = I
    W = X * S;
    grad_h1 = -1*X.diagonal().asDiagonal().inverse();
    grad_h1 += W;
    h1 = - X.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(X.transpose())*W).trace();

    int outer_itr_count, inner_itr_count;        // iteration counts

    outer_itr_count = 0;
    while (true) {

        clock_t start = clock();
 
        tau = tau_;

        inner_itr_count = 0;
        while (true) {

            tmp = MatrixXd(X) - tau*grad_h1;
            sthreshmat(tmp, tau, LambdaMat);
            Xn = tmp.sparseView();

            if (tmp.diagonal().minCoeff() > 0) {

                Step = Xn - X;
                Wn = Xn * S;

                // h1n = - Xn.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(Xn.transpose()).cwiseProduct(Wn).sum());
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

        clock_t end = clock();
        double elapsed = double(end - start)/CLOCKS_PER_SEC;
        double fro_norm = (MatrixXd(Xn)-Omega_star).norm();

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
        // delta_subg = subg.norm()/Xn.norm();
        delta_updates = (Xn-X).norm();
        hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();

        hist_inner_itr_count(outer_itr_count) = inner_itr_count;
        // hist_delta_subg(outer_itr_count) = delta_subg;
        hist_delta_updates(outer_itr_count) = delta_updates;
        hist_hn(outer_itr_count) = hn;
        hist_norm(outer_itr_count) = fro_norm;
        hist_iter_time(outer_itr_count) = elapsed;

        if (delta_updates < epstol || outer_itr_count > maxitr) {
            
            if (outer_itr_count < maxitr) {
                hist_inner_itr_count(outer_itr_count+1) = -1;
                // hist_delta_subg(outer_itr_count+1) = -1;
                hist_delta_updates(outer_itr_count+1) = -1;
                hist_hn(outer_itr_count+1) = -1;
                hist_norm(outer_itr_count+1) = -1;
                hist_iter_time(outer_itr_count+1) = -1;
            }

            break;
        } else {
            h1 = h1n;
            X = Xn;
            W = Wn;
        }

        outer_itr_count += 1;

    }
    return Xn;
}

SparseMatrix<double> cce_constant(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S, 
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> Omega_star,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diagonal,
    Ref<VectorXd> hist_delta_updates,
    Ref<VectorXd> hist_hn,
    Ref<VectorXd> hist_norm,
    Ref<VectorXd> hist_iter_time
    ) {

    int p = S.cols();

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // X*S

    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd tmp(p, p);
    ArrayXd y(p);

    double hn, h1, h1n, delta_updates;

    X.setIdentity();            // initial guess: X = I
    grad_h1 = X * S;
    h1 = - X.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(X.transpose())*W).trace();

    int itr_count;        // iteration counts

    itr_count = 0;
    while (true) {
        
        clock_t start = clock();

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

        clock_t end = clock();
        double elapsed = double(end - start)/CLOCKS_PER_SEC;
        double fro_norm = (MatrixXd(Xn)-Omega_star).norm();
       
        Wn = Xn * S;
        Step = Xn - X;
        h1n = - Xn.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(Xn.transpose())*Wn).trace();
        grad_h1 = Wn;

        delta_updates = Step.norm();
        hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();

        hist_delta_updates(itr_count) = delta_updates;
        hist_hn(itr_count) = hn;
        hist_norm(itr_count) = fro_norm;
        hist_iter_time(itr_count) = elapsed;

        if (delta_updates < epstol || itr_count > maxitr) {
            if (itr_count < maxitr) {
                hist_delta_updates(itr_count+1) = -1;
                hist_hn(itr_count+1) = -1;
                hist_norm(itr_count+1) = -1;
                hist_iter_time(itr_count+1) = -1;
            }
            break;
        } else {
            h1 = h1n;
            X = Xn;
            W = Wn;
        }

        itr_count += 1;

    }
    return Xn;
}

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