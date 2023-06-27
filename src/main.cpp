#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/SparseCore>

namespace py = pybind11;

Eigen::SparseMatrix<double> ccista_backtracking(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    Eigen::Ref<Eigen::VectorXi> hist_inner_itr_count,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time
    );

Eigen::SparseMatrix<double> ccista_constant(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time
    );

Eigen::SparseMatrix<double> accord_backtracking(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Eigen::Ref<Eigen::VectorXi> hist_inner_itr_count,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time
    );

Eigen::SparseMatrix<double> accord_constant(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time
    );

PYBIND11_MODULE(_gaccord, m) {

    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: gaccord

        .. autosummary::
           :toctree: _generate

           add
           subtract
           det
           inv
           mutate
           incr_vector
           incr_matrix
           incr_matrix_any
    )pbdoc";

    m.def(
        "ccista_backtracking",
        &ccista_backtracking,
        R"pbdoc(
            CC-ISTA with backtracking

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "ccista_constant",
        &ccista_constant,
        R"pbdoc(
            CC-ISTA with constant step size

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "accord_backtracking",
        &accord_backtracking,
        R"pbdoc(
            ACCORD with backtracking

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "accord_constant",
        &accord_constant,
        R"pbdoc(
            ACCORD with constant step size

            Column-major sparse matrix
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
