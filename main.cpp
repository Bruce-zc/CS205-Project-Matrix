#include <iostream>
#include "Matrix.cpp"
#include <complex>
#include <ctime>

using namespace std;

int main()
{
     cout << "Q1Q2 - Test 0: Sparse matrix:" << endl;
     SparseMatrix<double> sm1(4, 4);
     cout << "The spare matrix sm1 is: " << sm1 << endl;
     sm1.insert(1, 1, 2);
     cout << "Insert a element (1,1), with value 2 : " << sm1 << endl;
     sm1.insert(1, 1, 1);
     cout << "Update a element (1,1), with value 1 : " << sm1 << endl;
     sm1.insert(0, 3, 5);
     sm1.insert(2, 1, 7);
     sm1.insert(2, 2, 4);
     cout << "Insert 3 more elements: " << sm1 << endl;

     sm1.insert(Trituple<double>{3, 1, 4});
     cout << "Q1Q2 - Exception 0: Out of Bound:" << endl;
     sm1.insert(100, 100, 1);

     cout << "Q1Q2 - Test 0: Convert sparse matrix to matrix:" << endl;
     cout << "Equivalent normal matrix:" << endl;
     Matrix<double> m1(sm1);
     cout << m1 << endl
          << endl;

     cout << "Q1Q2 - Test 1: Initialize 2 double matrices:" << endl;
     cout << "Matrix 1 (4 x 4 double matrix):" << endl;
     vector<vector<double>> vec1 = {{-0.4, 2.1, 3.7, -4}, {5.8, 1.2, 0.7, -0.8}, {1.9, 1.1, -1, 2}, {-4.3, 4.1, 5.5, 6.2}};
     Matrix<double> matrix1(vec1);
     cout << "Matrix1 = \n"
          << matrix1 << endl
          << endl;

     cout << "Matrix 2 (4 x 4 double matrix):" << endl;
     vector<vector<double>> vec2 = {{1.15, 2.21, -0.3, 4}, {-5, -6, -3.18, 0.8}, {0.9, 0.1, 0.1, 0.2}, {1.3, -1.4, -0.5, 2.6}};
     Matrix<double> matrix2(vec2);
     cout << "Matrix2 = \n"
          << matrix2 << endl
          << endl;

     cout << "Q1Q2 - Test 2: Initialize an int matrix:" << endl;
     cout << "Matrix 3 (4 x 4 int matrix):" << endl;
     vector<vector<int>> vec3 = {{1, 1, 8, 1}, {2, 2, 1, 4}, {1, 1, 8, 1}, {3, 2, 1, 9}};
     Matrix<int> matrix3(vec3);
     cout << "Matrix3 = \n"
          << matrix3 << endl
          << endl;

     cout << "Q1Q2 - Test 3: Initialize a complex double matrix:" << endl;
     cout << "Matrix 4 (4 x 4 complex double matrix):" << endl;
     vector<vector<complex<double>>> vec4 = {{complex<double>(1, 2), -3, 1.1}, {-5, 1, complex<double>(0, 2)}, {complex<double>(9, 2), 3, 0}};
     Matrix<complex<double>> matrix4(vec4);
     cout << "Matrix4 = \n"
          << matrix4 << endl
          << endl;
     cout << "Q3 - Test 1: Add 2 matrices:" << endl;
     cout << "Matrix1 + Matrix2 = \n"
          << (matrix1 + matrix2) << endl
          << endl;

     cout << "Q3 - Test 2: Subtract 2 matrices:" << endl;
     cout << "Matrix1 - Matrix2 = \n"
          << (matrix1 - matrix2) << endl
          << endl;

     cout << "Q9 - Exception 1: Add/Subtract 2 matrices, size mismatch:" << endl;
     cout << "Matrix 5 (3 x 3 double matrix):" << endl;
     vector<vector<double>> vec5 = {{1.15, -0.3, 4.1}, {-5.2, 1.6, -3.18}, {-0.1, 0.7, 0.2}};
     Matrix<double> matrix5(vec5);
     cout << "Matrix5 = \n"
          << matrix5 << endl
          << endl;
     cout << "Matrix1 - Matrix5 = \n"
          << (matrix1 - matrix5) << endl
          << endl;

     cout << "Q3 - Test 3: Scalar multiplicayion:" << endl;
     cout << "Matrix1 * 5 = \n"
          << (matrix1 * 5) << endl
          << endl;

     cout << "Q3 - Test 4: Scalar division:" << endl;
     cout << "Matrix1 / 5 = \n"
          << (matrix1 / 5) << endl
          << endl;

     cout << "Q9 - Exception 2: Scalar division, divided by 0:" << endl;
     cout << "Matrix1 / 0 = \n"
          << (matrix1 / 0) << endl
          << endl;

     cout << "Q3 - Test 5: Transpose:" << endl;
     cout << "transpose(Matrix4) = \n"
          << matrix4.transpose() << endl
          << endl;

     cout << "Q3 - Test 6: Conjugate:" << endl;
     cout << "conjugate(Matrix4) = \n"
          << matrix4.conjugate() << endl
          << endl;

     cout << "Q3 - Test 7: Elementwise multiplication:" << endl;
     cout << "Matrix1 .* Matrix2 = \n"
          << matrix1.elementwise_multiply(matrix2) << endl
          << endl;

     cout << "Q9 - Exception 3: Elementwise Mmultiplication, size mismatch:" << endl;
     cout << "Matrix1 .* Matrix5 = \n"
          << matrix1.elementwise_multiply(matrix5) << endl
          << endl;

     cout << "Q3 - Test 8: Multiply 2 matrices:" << endl;
     cout << "Matrix1 * Matrix2 = \n"
          << (matrix1 * matrix2) << endl
          << endl;

     cout << "Q3 - Test 9: Dot product:" << endl;
     cout << "Column Vector (4 x 1 double matrix):" << endl;
     vector<vector<double>> vec6 = {{1.1}, {2.1}, {3.1}, {4.1}};
     Matrix<double> col_vec(vec6);
     cout << "Column Vector = \n"
          << col_vec << endl
          << endl;
     cout << "Row Vector (1 x 4 double matrix):" << endl;
     vector<vector<double>> vec7 = {{1.1, 2.1, 3.1, 4.1}};
     Matrix<double> row_vec(vec7);
     cout << "Row Vector = \n"
          << row_vec << endl
          << endl;
     cout << "Column Vector dot Row Vector = \n"
          << (col_vec.dot(row_vec)) << endl
          << endl;

     cout << "Q3 - Test 10: Cross product:" << endl;
     cout << "Column Vector 1 (3 x 1 int matrix):" << endl;
     vector<vector<int>> vec8 = {{0}, {1}, {0}};
     Matrix<int> col_vec1(vec8);
     cout << "Column Vector 1 = \n"
          << col_vec1 << endl
          << endl;
     cout << "Column Vector 2 (3 x 1 int matrix):" << endl;
     vector<vector<int>> vec9 = {{1}, {0}, {0}};
     Matrix<int> col_vec2(vec9);
     cout << "Column Vector 2 = \n"
          << col_vec2 << endl
          << endl;
     cout << "Column Vector 1 cross product Column Vector 2 = \n"
          << (col_vec1.cross(col_vec2)) << endl
          << endl;

     cout << "Q9 - Exception 4: Cross product, not supported:" << endl;
     cout << "Column Vector 1 cross product (Column Vector 2)T = \n"
          << col_vec1.cross(col_vec2.transpose()) << endl
          << endl;

     cout << "Q4 - Test 1: Max:" << endl;
     cout << "Matrix1 = \n"
          << matrix1 << endl
          << endl;
     cout << "max of Matrix1 = \n"
          << matrix1.max() << endl
          << endl;
     cout << "max of Matrix1 by column = \n"
          << matrix1.max(0) << endl
          << endl;
     cout << "max of Matrix1 by row = \n"
          << matrix1.max(1) << endl
          << endl;
     cout << "(Bonus-1) max of Matrix1 by row (keepdims = true) = \n"
          << matrix1.max(1, true) << endl
          << endl;

     cout << "Q4 - Test 2: Min:" << endl;
     cout << "Matrix1 = \n"
          << matrix1 << endl
          << endl;
     cout << "min of Matrix1 = \n"
          << matrix1.min() << endl
          << endl;
     cout << "min of Matrix1 by column = \n"
          << matrix1.min(0) << endl
          << endl;
     cout << "min of Matrix1 by row = \n"
          << matrix1.min(1) << endl
          << endl;
     cout << "(Bonus-1) min of Matrix1 by row (keepdims = true) = \n"
          << matrix1.min(1, true) << endl
          << endl;

     cout << "Q4 - Test 3: Sum:" << endl;
     cout << "Matrix1 = \n"
          << matrix1 << endl
          << endl;
     cout << "sum of Matrix1 = \n"
          << matrix1.sum() << endl
          << endl;
     cout << "sum of Matrix1 by column = \n"
          << matrix1.sum(0) << endl
          << endl;
     cout << "sum of Matrix1 by row = \n"
          << matrix1.sum(1) << endl
          << endl;
     cout << "(Bonus-1) sum of Matrix1 by row (keepdims = true) = \n"
          << matrix1.sum(1, true) << endl
          << endl;

     cout << "Q4 - Test 4: Avg:" << endl;
     cout << "Matrix1 = \n"
          << matrix1 << endl
          << endl;
     cout << "avg of Matrix1 = \n"
          << matrix1.avg() << endl
          << endl;
     cout << "avg of Matrix1 by column = \n"
          << matrix1.avg(0) << endl
          << endl;
     cout << "avg of Matrix1 by row = \n"
          << matrix1.avg(1) << endl
          << endl;
     cout << "(Bonus-1) avg of Matrix1 by row (keepdims = true) = \n"
          << matrix1.avg(1, true) << endl
          << endl;

     cout << "Q9 - Exception 5: Max/min/sum on wrong dimension:" << endl;
     cout << "max of Matrix1 by axis 2 = \n"
          << matrix1.max(2) << endl
          << endl;
     cout << "min of Matrix1 by axis 2 = \n"
          << matrix1.min(-1) << endl
          << endl;
     cout << "sum of Matrix1 by axis 2 = \n"
          << matrix1.sum(7) << endl
          << endl;
     cout << "avg of Matrix1 by axis 2 = \n"
          << matrix1.avg(-3.14) << endl
          << endl;

     cout << "Q6 - Test 1: Slice:" << endl;
     cout << "Matrix1 = \n"
          << matrix1 << endl
          << endl;
     cout << "Matrix1[1:2, 1:2] = \n"
          << matrix1.slice(1, 2, 1, 2) << endl
          << endl;
     cout << "Matrix1[1:4, 1:4] = \n"
          << matrix1.slice(1, 4, 1, 4) << endl
          << endl;
     cout << "Matrix1[0:3, 1:3] = \n"
          << matrix1.slice(0, 3, 1, 3) << endl
          << endl;

     cout << "Q9 - Exception 6: Slice with wrong input:" << endl;
     cout << "Matrix1[1:5, 1:5] = \n"
          << matrix1.slice(1, 5, 1, 5) << endl
          << endl;
     cout << "Matrix1[2:1, 2:1] = \n"
          << matrix1.slice(2, 1, 2, 1) << endl
          << endl;

     cout << "Q6 - Test 2: Reshape:" << endl;
     cout << "Matrix1 = \n"
          << matrix1 << endl
          << endl;
     cout << "Matrix1.reshape(2, 8) = \n"
          << matrix1.reshape(2, 8) << endl
          << endl;
     cout << "(Bonus-2) Matrix1.reshape(2, 8), column first = \n"
          << matrix1.reshape(2, 8, true) << endl
          << endl;
     cout << "(Bonus-2) Matrix1.reshape(8, -1) = \n"
          << matrix1.reshape(8, -1) << endl
          << endl;
     cout << "(Bonus-2) Matrix1.reshape(-1, 1) = \n"
          << matrix1.reshape(-1, 1, true) << endl
          << endl;

     cout << "Q9 - Exception 7: Reshape with wrong input:" << endl;
     cout << "Matrix1.reshape(3, 7) = \n"
          << matrix1.reshape(3, 7) << endl
          << endl;
     cout << "Matrix1.reshape(3, -1) = \n"
          << matrix1.reshape(3, -1) << endl
          << endl;
     cout << "Matrix1.reshape(-1, -1) = \n"
          << matrix1.reshape(-1, -1) << endl
          << endl;

     cout << "Q5 - Test 1: Eigenvalue:" << endl;
     vector<vector<double>> vec_eigenvalue_1 = {
         {-0.4, 2.1, 3.7, -4},
         {5.8, 1.2, 0.7, -0.8},
         {1.9, 1.1, -1, 2},
         {-4.3, 4.1, 5.5, 6.2}};
     Matrix<double> matrix_eigenvalue_1(vec_eigenvalue_1);
     cout << "matrix_eigenvalue_1 is:" << endl
          << matrix_eigenvalue_1 << endl;
     cout << "The eigenvalues of matrix_eigenvalue_1 are: \n"
          << matrix_eigenvalue_1.eigenvalue() << endl;

     vector<vector<double>> vec_eigenvalue_2 = {
         {-2, 1, 1},
         {0, 2, 0},
         {-4, 1, 3}};
     Matrix<double> matrix_eigenvalue_2(vec_eigenvalue_2);
     cout << "matrix_eigenvalue_2 is:" << endl
          << matrix_eigenvalue_2 << endl;
     cout << "The eigenvalues of matrix_eigenvalue_2 are: \n"
          << matrix_eigenvalue_2.eigenvalue() << endl;

     vector<vector<double>> vec_eigenvalue_3 = {
         {-2, 1, 1},
         {0, 2, 0}};
     Matrix<double> matrix_eigenvalue_3(vec_eigenvalue_3);
     cout << "matrix_eigenvalue_3 is:" << endl
          << matrix_eigenvalue_3 << endl;
     cout << "The eigenvalues of matrix_eigenvalue_3 are: \n"
          << matrix_eigenvalue_3.eigenvalue() << endl;

     cout << "Q5 - Test 2: Eigenvector:" << endl;
     vector<vector<double>> vec_eigenvector_1 = {
         {-0.4, 2.1, 3.7, -4},
         {5.8, 1.2, 0.7, -0.8},
         {1.9, 1.1, -1, 2},
         {-4.3, 4.1, 5.5, 6.2}};
     Matrix<double> matrix_eigenvector_1(vec_eigenvector_1);
     cout << "matrix_eigenvector_1 is:" << endl
          << matrix_eigenvector_1 << endl;
     cout << "The eigenvectors of matrix_eigenvector_1 are: \n"
          << matrix_eigenvector_1.eigenvector() << endl;

     vector<vector<double>> vec_eigenvector_2 = {
         {-2, 1, 1},
         {0, 2, 0},
         {-4, 1, 3}};
     Matrix<double> matrix_eigenvector_2(vec_eigenvector_2);
     cout << "matrix_eigenvector_2 is:" << endl
          << matrix_eigenvector_2 << endl;
     cout << "The eigenvectors of matrix_eigenvector_2 are: \n"
          << matrix_eigenvector_2.eigenvector() << endl;

     vector<vector<double>> vec_eigenvector_3 = {
         {-2, 1, 1},
         {0, 2, 0}};
     Matrix<double> matrix_eigenvector_3(vec_eigenvector_3);
     cout << "matrix_eigenvector_3 is:" << endl
          << matrix_eigenvector_3 << endl;
     cout << "The eigenvectors of matrix_eigenvector_3 are: \n"
          << matrix_eigenvector_3.eigenvector() << endl;

     cout << "Q5 - Test 3: trace:" << endl;
     vector<vector<double>> vec_trace_1 = {
         {-0.4, 2.1, 3.7, -4},
         {5.8, 1.2, 0.7, -0.8},
         {1.9, 1.1, -1, 2},
         {-4.3, 4.1, 5.5, 6.2}};
     Matrix<double> matrix_trace_1(vec_trace_1);
     cout << "matrix_trace_1 is:" << endl
          << matrix_trace_1 << endl;
     cout << "The trace of matrix_trace_1 are: \n"
          << matrix_trace_1.trace() << endl;

     vector<vector<double>> vec_trace_2 = {
         {-2, 1, 1},
         {0, 2, 0},
         {-4, 1, 3}};
     Matrix<double> matrix_trace_2(vec_trace_2);
     cout << "matrix_trace_2 is:" << endl
          << matrix_trace_2 << endl;
     cout << "The trace of matrix_trace_2 is: \n"
          << matrix_trace_2.trace() << endl;

     vector<vector<double>> vec_trace_3 = {
         {-2, 1, 1},
         {0, 2, 0}};
     Matrix<double> matrix_trace_3(vec_trace_3);
     cout << "matrix_trace_3 is:" << endl
          << matrix_trace_3 << endl;
     cout << "The trace of matrix_trace_3 is: \n"
          << matrix_trace_3.trace() << endl;

     cout << "Q5 - Test 4: inverse:" << endl;
     vector<vector<double>> vec_inverse_1 = {
         {-0.4, 2.1, 3.7, -4},
         {5.8, 1.2, 0.7, -0.8},
         {1.9, 1.1, -1, 2},
         {-4.3, 4.1, 5.5, 6.2}};
     Matrix<double> matrix_inverse_1(vec_inverse_1);
     cout << "matrix_inverse_1 is:" << endl
          << matrix_inverse_1 << endl;
     cout << "The inverse of matrix_inverse_1 is: \n"
          << matrix_inverse_1.inverse() << endl;

     vector<vector<double>> vec_inverse_2 = {
         {1, 2, 3},
         {2, 5, 3},
         {1, 0, 8}};
     Matrix<double> matrix_inverse_2(vec_inverse_2);
     cout << "matrix_inverse_2 is:" << endl
          << matrix_inverse_2 << endl;
     cout << "The inverse of matrix_inverse_2 is: \n"
          << matrix_inverse_2.inverse() << endl;

     vector<vector<double>> vec_inverse_3 = {
         {1, 0, 0},
         {0, 1, 0},
         {0, 0, 1}};
     Matrix<double> matrix_inverse_3(vec_inverse_3);
     cout << "matrix_inverse_3 is:" << endl
          << matrix_inverse_3 << endl;
     cout << "The inverse of matrix_inverse_3 is: \n"
          << matrix_inverse_3.inverse() << endl;

     vector<vector<double>> vec_inverse_4 = {
         {-2, 1, 1},
         {0, 2, 0}};
     Matrix<double> matrix_inverse_4(vec_inverse_4);
     cout << "matrix_inverse_4 is:" << endl
          << matrix_inverse_4 << endl;
     cout << "The inverse of matrix_inverse_4 is: \n"
          << matrix_inverse_4.inverse() << endl;

     vector<vector<double>> vec_inverse_5 = {
         {1, 0, 1},
         {0, 0, 0},
         {0, 0, 1}};
     Matrix<double> matrix_inverse_5(vec_inverse_5);
     cout << "matrix_inverse_5 is:" << endl
          << matrix_inverse_5 << endl;
     cout << "The inverse of matrix_inverse_5 is: \n"
          << matrix_inverse_5.inverse() << endl;

     cout << "Q5 - Test 5: determinent:" << endl;
     vector<vector<double>> vec_det_1 = {
         {-0.4, 2.1, 3.7, -4},
         {5.8, 1.2, 0.7, -0.8},
         {1.9, 1.1, -1, 2},
         {-4.3, 4.1, 5.5, 6.2}};
     Matrix<double> matrix_det_1(vec_det_1);
     cout << "matrix_det_1 is:" << endl
          << matrix_det_1 << endl;
     cout << "The det of matrix_det_1 is: \n"
          << matrix_det_1.det() << endl;

     vector<vector<double>> vec_det_2 = {
         {1, 2, 3},
         {2, 5, 3},
         {1, 0, 8}};
     Matrix<double> matrix_det_2(vec_det_2);
     cout << "matrix_det_2 is:" << endl
          << matrix_det_2 << endl;
     cout << "The det of matrix_det_2 is: \n"
          << matrix_det_2.det() << endl;

     vector<vector<double>> vec_det_3 = {
         {1, 0, 0},
         {0, 1, 0},
         {0, 0, 1}};
     Matrix<double> matrix_det_3(vec_det_3);
     cout << "matrix_det_3 is:" << endl
          << matrix_det_3 << endl;
     cout << "The det of matrix_det_3 is: \n"
          << matrix_det_3.det() << endl;

     vector<vector<double>> vec_det_4 = {
         {0, 1, 0},
         {0, 1, 1},
         {0, 0, 1}};
     Matrix<double> matrix_det_4(vec_det_4);
     cout << "matrix_det_4 is:" << endl
          << matrix_det_4 << endl;
     cout << "The det of matrix_det_4 is: \n"
          << matrix_det_4.det() << endl;

     cout << "Q7 - Test 1: same mode:" << endl;

     vector<vector<double>> vec_kernel_1 = {
         {-1, -2, -1},
         {0, 0, 0},
         {1, 2, 1}};
     Matrix<double> matrix_kernel_1(vec_kernel_1);

     vector<vector<double>> vec_conv_1 = {
         {1, 2, 3, 4},
         {5, 6, 7, 8},
         {9, 10, 11, 12},
         {13, 14, 15, 16}};
     Matrix<double> matrix_conv_1(vec_conv_1);

     Matrix<double> ans = matrix_conv_1.conv(matrix_kernel_1, 0);
     cout << ans << endl;
     cout << "Q7 - Test 2: full mode:" << endl;
     ans = matrix_conv_1.conv(matrix_kernel_1, 1);
     cout << ans << endl;
     cout << "Q7 - Test 3: valid mode:" << endl;
     ans = matrix_conv_1.conv(matrix_kernel_1, 2);
     cout << ans << endl;
     return 0;
}
