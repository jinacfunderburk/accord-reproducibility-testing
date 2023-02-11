#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/SparseCore>

namespace py = pybind11;

Eigen::SparseMatrix<double> cce(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diagonal,
    Eigen::Ref<Eigen::VectorXd> hist_norm_diff,
    Eigen::Ref<Eigen::VectorXd> hist_hn
    );

PYBIND11_MODULE(_gconcorde, m) {

    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: gconcorde

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
        "cce",
        &cce,
        R"pbdoc(
            CCE

            Column-major sparse matrix
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
