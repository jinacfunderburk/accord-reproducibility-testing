#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/SparseCore>


namespace py = pybind11;

// https://github.com/pybind/pybind11/blob/479e9a50f38cfa8e525a697d7f5d850f851223b5/tests/test_eigen.cpp#L120

// // simple declared function
// int add(int i, int j);
// 
// // return a double float computed from immutable input matrix
// double det(const Eigen::MatrixXd &xs);
// 
// // return a matrix derived from input matrix
// Eigen::MatrixXd inv(const Eigen::MatrixXd &xs);
// 
// // modify input matrix in place
// void mutate(Eigen::Ref<Eigen::MatrixXd> x);
// 
// // modify input vector in place and return reference
// Eigen::Ref<Eigen::VectorXd> incr_vector(Eigen::Ref<Eigen::VectorXd> m, double v);
// 
// // modify input matrix in place (requires order="F")
// Eigen::Ref<Eigen::MatrixXd> incr_matrix(Eigen::Ref<Eigen::MatrixXd> m, double v);
// 
// // modify input matrix in place (any storage order)
// // see https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html?highlight=eigendref#storage-orders
// // py::EigenDRef<Eigen::MatrixXd> incr_matrix_any(py::EigenDRef<Eigen::MatrixXd> m, double v); // short version
// Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> incr_matrix_any(Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> m, double v); // long version
// 
// Eigen::SparseMatrix<float> ccista(Eigen::Ref<Eigen::SparseMatrix<float> > xs);
// Eigen::SparseMatrix<float> ccista(Eigen::Ref<Eigen::MatrixXf> S);
// Eigen::SparseMatrix<double> ccista(
Eigen::SparseMatrix<double> ccista(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double epstol,
    int maxitr,
    Eigen::Ref<Eigen::VectorXi> hist_inner_itr_count,
    Eigen::Ref<Eigen::VectorXd> hist_delta_subg,
    Eigen::Ref<Eigen::VectorXd> hist_hn
    );

Eigen::SparseMatrix<double> cceista(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    double epstol,
    int maxitr,
    Eigen::Ref<Eigen::VectorXi> hist_inner_itr_count,
    Eigen::Ref<Eigen::VectorXd> hist_delta_updates,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time
    );

Eigen::SparseMatrix<double> cce_constant(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diagonal,
    Eigen::Ref<Eigen::VectorXd> hist_delta_updates,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time
    );
// Eigen::SparseMatrix<double> ccista(
//     Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
//     Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
//     double epstol,
//     int maxitr,
//     Eigen::Ref<Eigen::VectorXi> hist_inner_itr_count,
//     Eigen::Ref<Eigen::VectorXd> hist_delta_subg,
//     Eigen::Ref<Eigen::VectorXd> hist_hn
//     );
// 
// Eigen::SparseMatrix<float, Eigen::RowMajor> sparsify_r(Eigen::MatrixXf &xs);

PYBIND11_MODULE(_gconcorde, m) {

    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: gconcord

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
        "ccista",
        &ccista,
        R"pbdoc(
            CC-ISTA

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "cceista",
        &cceista,
        R"pbdoc(
            CCE-ISTA

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "cce_constant",
        &cce_constant,
        R"pbdoc(
            CCE-Constant

            Column-major sparse matrix
        )pbdoc");


    // // inline function
    // m.def(
    //     "subtract", 
    //     [](int i, int j) { return i - j; }, 
    //     R"pbdoc(
    //         Subtract two numbers

    //         Test of an inline function
    //     )pbdoc");

    // m.def(
    //     "add", 
    //     &add, 
    //     R"pbdoc(
    //         Add two numbers

    //         Test of a delcared function
    //     )pbdoc");

    // m.def(
    //     "inv", 
    //     &inv,
    //     R"pbdoc(
    //         Add two numbers

    //         Test of a delcared function
    //     )pbdoc");

    // m.def("det", &det);

    // m.def(
    //     "mutate", 
    //     &mutate,
    //     R"pbdoc(
    //         Mutate matrix

    //         A simple example of modifying the input matrix in place.
    //     )pbdoc");

    // m.def(
    //     "incr_vector",
    //     &incr_vector,
    //     py::return_value_policy::reference,
    //     R"pbdoc(
    //         Increment vector
    //         
    //         Modify input vector in place by adding a constant value to every element.
    //         Return a reference to the input vector
    //     )pbdoc");

    // m.def(
    //     "incr_matrix",
    //     &incr_matrix,
    //     py::return_value_policy::reference,
    //     R"pbdoc(
    //         Increment matrix
    //         
    //         Modify input matrix in place by adding a constant value to every element.
    //         Return a reference to the input matrix

    //         Requires order=\"F\"
    //     )pbdoc");

    // m.def(
    //     "incr_matrix_any",
    //     &incr_matrix_any,
    //     py::return_value_policy::reference,
    //     R"pbdoc(
    //         Increment matrix (any storage order)
    //         
    //         Modify input matrix in place by adding a constant value to every element.
    //         Return a reference to the input matrix

    //         Any storage order works
    //     )pbdoc");

    // m.def(
    //     "sparsify_c",
    //     &sparsify_c,
    //     py::return_value_policy::reference,
    //     R"pbdoc(
    //         Create sparse version of input matrix

    //         Column-major sparse matrix
    //     )pbdoc");

    // m.def(
    //     "sparsify_r",
    //     &sparsify_r,
    //     py::return_value_policy::reference,
    //     R"pbdoc(
    //         Create sparse version of input matrix

    //         Row-major sparse matrix
    //     )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
