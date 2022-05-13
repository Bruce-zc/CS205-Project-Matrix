#include <iostream>
#include "Matrix.cpp"
#include <complex>
#include <ctime>

using namespace std;

int main()
{
    cout << "Q1Q2 - Test 1: Initialize 2 double matrices:" << endl;
    cout << "Matrix 1 (4 x 4 double matrix):" << endl;
    vector<vector<double>> vec1 = {{-0.4, 2.1, 3.7, -4}, {5.8, 1.2, 0.7, -0.8}, {1.9, 1.1, -1, 2}, {-4.3, 4.1, 5.5, 6.2}};
    Matrix<double> matrix1(vec1);
    cout << "Matrix1 = \n" << matrix1 << endl << endl;
    cout << "Matrix 2 (4 x 4 double matrix):" << endl;
    vector<vector<double>> vec2 = {{1.15, 2.21, -0.3, 4}, {-5, -6, -3.18, 0.8}, {0.9, 0.1, 0.1, 0.2}, {1.3, -1.4, -0.5, 2.6}};
    Matrix<double> matrix2(vec2);
    cout << "Matrix2 = \n"  << matrix2 << endl << endl;

    cout << "Q1Q2 - Test 2: Initialize an int matrix:" << endl;
    cout << "Matrix 3 (4 x 4 int matrix):" << endl;
    vector<vector<int>> vec3 = {{1, 1, 8, 1}, {2, 2, 1, 4}, {1, 1, 8, 1}, {3, 2, 1, 9}};
    Matrix<int> matrix3(vec3);
    cout << "Matrix3 = \n"  << matrix3 << endl << endl;

    cout << "Q1Q2 - Test 3: Initialize a complex double matrix:" << endl;
    cout << "Matrix 4 (4 x 4 complex double matrix):" << endl;
    vector<vector<complex<double>>> vec4 = {{complex<double>(1, 2), -3, 1.1}, {-5, 1, complex<double>(0, 2)}, {complex<double>(9, 2), 3, 0}};
    Matrix<complex<double>> matrix4(vec4);
    cout << "Matrix4 = \n"  << matrix4 << endl << endl;

    cout << "Q3 - Test 1: Add 2 matrices:" << endl;
    cout << "Matrix1 + Matrix2 = \n" << (matrix1 + matrix2) << endl << endl;

    cout << "Q3 - Test 2: Subtract 2 matrices:" << endl;
    cout << "Matrix1 - Matrix2 = \n" << (matrix1 - matrix2) << endl << endl;

    cout << "Q9 - Exception 1: Add/Subtract 2 matrices, size mismatch:" << endl;
    cout << "Matrix 5 (3 x 3 double matrix):" << endl;
    vector<vector<double>> vec5 = {{1.15, -0.3, 4.1}, {-5.2, 1.6, -3.18}, {-0.1, 0.7, 0.2}};
    Matrix<double> matrix5(vec5);
    cout << "Matrix5 = \n"  << matrix5 << endl << endl;
    cout << "Matrix1 - Matrix5 = \n" << (matrix1 - matrix5) << endl << endl;

    cout << "Q3 - Test 3: Scalar multiplicayion:" << endl;
    cout << "Matrix1 * 5 = \n" << (matrix1 * 5) << endl << endl;

    cout << "Q3 - Test 4: Scalar division:" << endl;
    cout << "Matrix1 / 5 = \n" << (matrix1 / 5) << endl << endl;

    cout << "Q9 - Exception 2: Scalar division, divided by 0:" << endl;
    cout << "Matrix1 / 0 = \n" << (matrix1 / 0) << endl << endl;

    cout << "Q3 - Test 5: Transpose:" << endl;
    cout << "transpose(Matrix4) = \n" << matrix4.transpose() << endl << endl;

    cout << "Q3 - Test 6: Conjugate:" << endl;
    cout << "conjugate(Matrix4) = \n" << matrix4.conjugate() << endl << endl;

    cout << "Q3 - Test 7: Elementwise multiplication:" << endl;
    cout << "Matrix1 .* Matrix2 = \n" << matrix1.elementwise_multiply(matrix2) << endl << endl;

    cout << "Q9 - Exception 3: Elementwise Mmultiplication, size mismatch:" << endl;
    cout << "Matrix1 .* Matrix5 = \n" << matrix1.elementwise_multiply(matrix5) << endl << endl;

    cout << "Q3 - Test 8: Multiply 2 matrices:" << endl;
    cout << "Matrix1 * Matrix2 = \n" << (matrix1 * matrix2) << endl << endl;

    cout << "Q3 - Test 9: Dot product:" << endl;
    cout << "Column Vector (4 x 1 double matrix):" << endl;
    vector<vector<double>> vec6 = {{1.1}, {2.1}, {3.1}, {4.1}};
    Matrix<double> col_vec(vec6);
    cout << "Column Vector = \n"  << col_vec << endl << endl;
    cout << "Row Vector (1 x 4 double matrix):" << endl;
    vector<vector<double>> vec7 = {{1.1, 2.1, 3.1, 4.1}};
    Matrix<double> row_vec(vec7);
    cout << "Row Vector = \n"  << row_vec << endl << endl;
    cout << "Column Vector dot Row Vector = \n" << (col_vec.dot(row_vec)) << endl << endl;

    cout << "Q3 - Test 10: Cross product:" << endl;
    cout << "Column Vector 1 (3 x 1 int matrix):" << endl;
    vector<vector<int>> vec8 = {{0}, {1}, {0}};
    Matrix<int> col_vec1(vec8);
    cout << "Column Vector 1 = \n"  << col_vec1 << endl << endl;
    cout << "Column Vector 2 (3 x 1 int matrix):" << endl;
    vector<vector<int>> vec9 = {{1}, {0}, {0}};
    Matrix<int> col_vec2(vec9);
    cout << "Column Vector 2 = \n"  << col_vec2 << endl << endl;
    cout << "Column Vector 1 cross product Column Vector 2 = \n" << (col_vec1.cross(col_vec2)) << endl << endl;

    cout << "Q9 - Exception 4: Cross product, not supported:" << endl;
    cout << "Column Vector 1 cross product (Column Vector 2)T = \n" << col_vec1.cross(col_vec2.transpose()) << endl << endl;

    return 0;
}