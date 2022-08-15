#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>

int add(int i, int j) {
    return i + j;
}

// matrix determinant
double det(const Eigen::MatrixXd &xs) {
  return xs.determinant();
}

// matrix inverse
Eigen::MatrixXd inv(const Eigen::MatrixXd &xs) {
  return xs.inverse();
}

// modify input matrix in place
void mutate(Eigen::Ref<Eigen::MatrixXd> x) {
    x(0, 0) = 30.201;
}

// modify input vector in place and return a reference to it
Eigen::Ref<Eigen::VectorXd> incr_vector(Eigen::Ref<Eigen::VectorXd> m, double v) {
    m += Eigen::VectorXd::Constant(m.rows(), v);
    return m;
}

// modify input matrix in place and return a reference to it
Eigen::Ref<Eigen::MatrixXd> incr_matrix(Eigen::Ref<Eigen::MatrixXd> m, double v) {
    m += Eigen::MatrixXd::Constant(m.rows(), m.cols(), v);
    return m;
}

// modify input matrix in place and return a reference to it
// see https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html?highlight=eigendref#storage-orders
// py::EigenDRef<Eigen::MatrixXd> incr_matrix_any(py::EigenDRef<Eigen::MatrixXd> m, double v); // short version
Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> incr_matrix_any(Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> m, double v) {
    m += Eigen::MatrixXd::Constant(m.rows(), m.cols(), v);
    return m;
}

Eigen::SparseMatrix<float> sparsify_c(Eigen::MatrixXf &xs) { 
    return Eigen::SparseView<Eigen::MatrixXf>(xs); 
}

Eigen::SparseMatrix<float, Eigen::RowMajor> sparsify_r(Eigen::MatrixXf &xs) { 
    return Eigen::SparseView<Eigen::MatrixXf>(xs); 
}