#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "Matrix.h"

using namespace std;
using namespace cv;

template <class T>
class Matrix
{
private:
    vector<vector<T>> matrix;
    int row, column;

public:
    /* Default constructor:
    -- It is called with no parameter.
    -- It sets row and column as zero. */
    Matrix() : row(0), column(0) { matrix.resize(0); }

    /* Constructor with given size:
    -- It is called with two parameters: (int) row and column.
    -- It sets row and column as given. */
    Matrix(int row, int column)
    {
        this->row = row;
        this->column = column;
        this->matrix.resize(row);
        for (int i = 0; i < row; i++)
        {
            this->matrix[i].resize(column);
        }
    }

    /* Constructor with a 2-D vector:
    -- It is called with one parameter: (vector<vector<T>>) vec.
    -- It directly sets the matrix as given. */
    Matrix(vector<vector<T>> vec)
    {
        this->row = vec.size();
        this->column = vec[0].size();
        this->matrix = vec;
    }

    /* Constructor with a matrix:
    -- Copy the matrix. */
    Matrix(Matrix const &other)
    {
        this->row = other.row;
        this->column = other.column;
        this->matrix = other.matrix;
    }

    // Two query functions to get the size of the private matrix.
    int getRow()
    {
        return row;
    }

    int getColumn()
    {
        return column;
    }

    /* Override operator [] to return a row vector.
    ! If the requested row is ou of bound, error information will be displayed. */
    vector<T> &operator[](int i)
    {
        if (i >= row)
        {
            cerr << "\033[31;1mIndex out of bound for row.\033[0m" << endl;
            abort();
        }
        return matrix[i];
    }

    // Override << to show the matrix;
    friend ostream &operator<<(ostream &ost, Matrix other)
    {
        ost << "[";
        for (int i = 0; i < other.row; i++)
        {
            if (i != 0)
            {
                ost << " ";
            }
            for (int j = 0; j < other.column; j++)
            {
                ost << other.matrix[i][j];
                if (j != other.column - 1)
                {
                    ost << ", ";
                }
            }
            if (i != other.row - 1)
            {
                ost << endl;
            }
        }
        ost << "]";

        return ost;
    }

    // Matrix addition
    Matrix operator+(Matrix other)
    {
        // Check if the two matrices are of the same size.
        if (other.getColumn() != column || other.getRow() != row)
        {
            cerr << "\033[31;1mCan't do + operation, the size of 2 matrices are not equal.\033[0m" << endl;
            return Matrix();
        }

        // Create a new matrix with the same size as the two matrices.
        Matrix answer(row, column);

        // Add the two matrices  elementwise.
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] + other.matrix[i][j];
            }
        }

        return answer;
    }

    // Matrix substraction
    Matrix operator-(Matrix other)
    {
        // Check if the two matrices are of the same size.
        if (other.getColumn() != column || other.getRow() != row)
        {
            cerr << "\033[31;1mCan't do - operation, the size of 2 matrices are not equal.\033[0m" << endl;
            return Matrix();
        }

        // Create a new matrix with the same size as the two matrices.
        Matrix answer(row, column);

        // Subtract the two matrices elementwise.
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] - other.matrix[i][j];
            }
        }

        return answer;
    }

    // Scalar multiplication
    Matrix operator*(T scalar)
    {
        // Create a new matrix with the same size as the matrix.
        Matrix answer(row, column);

        // Do scalar multiplication.
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] * scalar;
            }
        }

        return answer;
    }

    // Scalar division
    Matrix operator/(T scalar)
    {
        // Check if the scalar equals 0.
        if (scalar == 0)
        {
            cerr << "\033[31;1mCan't do / operation, can't be divided by 0.\033[0m" << endl;
            return Matrix();
        }

        // Create a new matrix with the same size as the matrix.
        Matrix answer(row, column);

        // Do scalar division.
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] / scalar;
            }
        }

        return answer;
    }

    // Transpose
    Matrix transpose()
    {
        Matrix answer(column, row);
        for (int i = 0; i < column; i++)
        {
            for (int j = 0; j < row; j++)
            {
                answer[i][j] = matrix[j][i];
            }
        }
        return answer;
    }

    // Conjugate
    Matrix conjugate()
    {
        Matrix answer(row, column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = conj(matrix[i][j]);
            }
        }
        return answer;
    }

    // Elementwise multiplication
    Matrix elementwise_multiply(Matrix other)
    {
        // Check if the two matrices are of the same size.
        if (other.getColumn() != column || other.getRow() != row)
        {
            cerr << "\033[31;1mCan't do elementwise_multiply operation, the size of 2 matrices are not equal.\033[0m" << endl;
            return Matrix();
        }

        // Create a new matrix with the same size as the two matrices.
        Matrix answer(row, column);

        // Subtract the two matrices elementwise.
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] * other.matrix[i][j];
            }
        }

        return answer;
    }

    // Matrix multiplication
    Matrix operator*(Matrix other)
    {
        if (column != other.row)
        {
            cerr << "\033[31;1mCan't do * operation, size mismatch.\033[0m" << endl;
            return Matrix();
        }
        Matrix answer(row, other.column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < other.column; j++)
            {
                for (int k = 0; k < column; k++)
                {
                    answer[i][j] += matrix[i][k] * other[k][j];
                }
            }
        }
        return answer;
    }

    // Dot product
    T dot(Matrix other)
    {
        if (column != 1 || other.row != 1)
        {
            cerr << "\033[31;1mCan't do dot product operation, only support vectors.\033[0m" << endl;
            return 0;
        }

        T answer = 0;
        for (int i = 0; i < row; i++)
        {
            answer += matrix[i][0] * other.matrix[0][i];
        }

        return answer;
    }

    // Cross product
    Matrix cross(Matrix other)
    {
        if (column == 1 && other.column == 1 && row == 2 && other.row == 2)
        {
            Matrix answer(1, 1);
            answer[0][0] = matrix[0][0] * other[1][0] - matrix[1][0] * other[0][0];
            return answer;
        }

        if (column == 1 && other.column == 1 && row == 3 && other.row == 3)
        {
            Matrix answer(3, 1);
            answer[0][0] = matrix[1][0] * other[2][0] - matrix[2][0] * other[1][0];
            answer[1][0] = matrix[2][0] * other[0][0] - matrix[0][0] * other[2][0];
            answer[2][0] = matrix[0][0] * other[1][0] - matrix[1][0] * other[0][0];
            return answer;
        }

        cerr << "\033[31;1mCan't do cross product operation, only support 2-dim and 3-dim column vectors.\033[0m" << endl;
        return Matrix();
    }

    // Max
    // Return the max element in the matrix.
    T max()
    {
        T max_value = matrix[0][0];
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if (matrix[i][j] > max_value)
                {
                    max_value = matrix[i][j];
                }
            }
        }
        return max_value;
    }

    // Return the max element by row or column.
    Matrix max(int axis, bool keepdims = false)
    {
        if (axis == 0)
        {
            Matrix answer(1, column);
            for (int i = 0; i < column; i++)
            {
                T max_value = matrix[0][i];
                for (int j = 0; j < row; j++)
                {
                    if (matrix[j][i] > max_value)
                    {
                        max_value = matrix[j][i];
                    }
                }
                answer[0][i] = max_value;
            }
            return answer;
        }
        else if (axis == 1)
        {
            Matrix answer(row, 1);
            for (int i = 0; i < row; i++)
            {
                T max_value = matrix[i][0];
                for (int j = 0; j < column; j++)
                {
                    if (matrix[i][j] > max_value)
                    {
                        max_value = matrix[i][j];
                    }
                }
                answer[i][0] = max_value;
            }
            if (keepdims)
            {
                return answer;
            }
            else
            {
                return answer.transpose();
            }
        }
        else
        {
            cerr << "\033[31;1mCan't do max operation on specified axis, only support axis 0 and 1.\033[0m" << endl;
            return Matrix();
        }
    }

    // Min
    // Return the min element in the matrix.
    T min()
    {
        T min_value = matrix[0][0];
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if (matrix[i][j] < min_value)
                {
                    min_value = matrix[i][j];
                }
            }
        }
        return min_value;
    }

    // Return the min element by row or column.
    Matrix min(int axis, bool keepdims = false)
    {
        if (axis == 0)
        {
            Matrix answer(1, column);
            for (int i = 0; i < column; i++)
            {
                T min_value = matrix[0][i];
                for (int j = 0; j < row; j++)
                {
                    if (matrix[j][i] < min_value)
                    {
                        min_value = matrix[j][i];
                    }
                }
                answer[0][i] = min_value;
            }
            return answer;
        }
        else if (axis == 1)
        {
            Matrix answer(row, 1);
            for (int i = 0; i < row; i++)
            {
                T min_value = matrix[i][0];
                for (int j = 0; j < column; j++)
                {
                    if (matrix[i][j] < min_value)
                    {
                        min_value = matrix[i][j];
                    }
                }
                answer[i][0] = min_value;
            }
            if (keepdims)
            {
                return answer;
            }
            else
            {
                return answer.transpose();
            }
        }
        else
        {
            cerr << "\033[31;1mCan't do min operation on specified axis, only support axis 0 and 1.\033[0m" << endl;
            return Matrix();
        }
    }

    // Sum
    // Return the sum of all elements in the matrix.
    T sum()
    {
        T sum_value = 0;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                sum_value += matrix[i][j];
            }
        }
        return sum_value;
    }

    // Return the max element by row or column.
    Matrix sum(int axis, bool keepdims = false)
    {
        if (axis == 0)
        {
            Matrix answer(1, column);
            for (int i = 0; i < column; i++)
            {
                T sum_value = 0;
                for (int j = 0; j < row; j++)
                {
                    sum_value += matrix[j][i];
                }
                answer[0][i] = sum_value;
            }
            return answer;
        }
        else if (axis == 1)
        {
            Matrix answer(row, 1);
            for (int i = 0; i < row; i++)
            {
                T sum_value = 0;
                for (int j = 0; j < column; j++)
                {
                    sum_value += matrix[i][j];
                }
                answer[i][0] = sum_value;
            }
            if (keepdims)
            {
                return answer;
            }
            else
            {
                return answer.transpose();
            }
        }
        else
        {
            cerr << "\033[31;1mCan't do sum operation on specified axis, only support axis 0 and 1.\033[0m" << endl;
            return Matrix();
        }
    }

    // Avg
    // Return the average element in the matrix.
    T avg()
    {
        return sum() / (row * column);
    }

    Matrix avg(int axis, bool keepdims = false)
    {
        if (axis == 0)
        {
            return sum(axis, keepdims) / row;
        }
        else if (axis == 1)
        {
            return sum(axis, keepdims) / column;
        }
        else
        {
            cerr << "\033[31;1mCan't do avg operation on specified axis, only support axis 0 and 1.\033[0m" << endl;
            return Matrix();
        }
    }

    Matrix slice(int start_row, int end_row, int start_col, int end_col)
    {
        if (start_row < 0 || start_col < 0 || start_row > row || start_col > column || end_row < 0 || end_col < 0 || end_row > row || end_col > column)
        {
            cerr << "\033[31;1mError, invalid input: index out of bound.\033[0m" << endl;
            return Matrix(0, 0);
        }

        if (start_row > end_row || start_col > end_col)
        {
            cerr << "\033[31;1mError, invalid input: end index greater than start index.\033[0m" << endl;
            return Matrix(0, 0);
        }

        Matrix answer(end_row - start_row, end_col - start_col);
        for (int i = 0; i < answer.row; i++)
        {
            for (int j = 0; j < answer.column; j++)
            {
                answer[i][j] = matrix[start_row + i][start_col + j];
            }
        }
        return answer;
    }

    Matrix reshape(int num_row, int num_col, bool column_first = false)
    {
        if (num_row == -1 && num_col != -1 && row * column % num_col == 0)
        {
            num_row = row * column / num_col;
        }
        else if (num_col == -1 && num_row != -1 && row * column % num_row == 0)
        {
            num_col = row * column / num_row;
        }
        else if (num_row == -1 && num_col == -1)
        {
            cerr << "\033[31;1mError, invalid input: please at least specify the size on one dimension.\033[0m" << endl;
            return Matrix(0, 0);
        }

        if (num_row * num_col != row * column)
        {
            cerr << "\033[31;1mError, invalid input: can't reshape to the specified shape.\033[0m" << endl;
            return Matrix(0, 0);
        }

        Matrix answer(num_row, num_col);
        for (int i = 0; i < answer.row; i++)
        {
            for (int j = 0; j < answer.column; j++)
            {
                if (!column_first)
                {
                    int index = i * answer.column + j;
                    answer[i][j] = matrix[index / column][index % column];
                }
                else
                {
                    int index = i + j * answer.row;
                    answer[i][j] = matrix[index % row][index / row];
                }
            }
        }
        return answer;
    }

    // Another constructor to support OpenCV Mat
    Matrix(cv::Mat CV_mat)
    {
        if (CV_mat.channels() != 1)
        {
            cerr << "\033[31;1mError, invalid input: only support 2-dim input (1 channel image).\033[0m" << endl;
            return;
        }
        row = CV_mat.rows;
        column = CV_mat.cols;
        this->matrix.resize(row);
        for (int i = 0; i < row; i++)
        {
            this->matrix[i].resize(column);
        }

        for (int i = 0; i < CV_mat.rows; i++)
        {
            for (int j = 0; j < CV_mat.cols; j++)
            {
                switch (CV_mat.type())
                {
                case CV_8U:
                    matrix[i][j] = CV_mat.at<uchar>(i, j);
                    break;
                case CV_8S:
                    matrix[i][j] = CV_mat.at<char>(i, j);
                    break;
                case CV_16U:
                    matrix[i][j] = CV_mat.at<ushort>(i, j);
                    break;
                case CV_16S:
                    matrix[i][j] = CV_mat.at<short>(i, j);
                    break;
                case CV_32S:
                    matrix[i][j] = CV_mat.at<int>(i, j);
                    break;
                case CV_32F:
                    matrix[i][j] = CV_mat.at<float>(i, j);
                    break;
                case CV_64F:
                    matrix[i][j] = CV_mat.at<double>(i, j);
                    break;
                case CV_16F:
                    matrix[i][j] = CV_mat.at<double>(i, j);
                    break;
                default:
                    break;
                }
            }
        }
    }

    // Matrix to mat
    cv::Mat toMat(int type = 0)
    {
        cv::Mat answer = cv::Mat::zeros(row, column, type);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                switch (type)
                {
                case CV_8U:
                    answer.at<uchar>(i, j) = matrix[i][j];
                    break;
                case CV_8S:
                    answer.at<char>(i, j) = matrix[i][j];
                    break;
                case CV_16U:
                    answer.at<ushort>(i, j) = matrix[i][j];
                    break;
                case CV_16S:
                    answer.at<short>(i, j) = matrix[i][j];
                    break;
                case CV_32S:
                    answer.at<int>(i, j) = matrix[i][j];
                    break;
                case CV_32F:
                    answer.at<float>(i, j) = matrix[i][j];
                    break;
                case CV_64F:
                    answer.at<double>(i, j) = matrix[i][j];
                    break;
                case CV_16F:
                    answer.at<double>(i, j) = matrix[i][j];
                    break;
                default:
                    break;
                }
            }
        }
        return answer;
    }

    Matrix eigenvalue() // Calculate eigenvalue (only real eigenvalue)
    {
        if (row != column)
        {
            cerr << "\033[31;1mMatrix must be a square matrix.\033[0m" << endl;
            return Matrix(this->row, 0);
        }
        Mat Mat_1 = toMat(CV_64F);
        Mat eigenvaluesMat;
        Mat eigenvectorsMat;
        eigenNonSymmetric(Mat_1, eigenvaluesMat, eigenvectorsMat);
        return Matrix(eigenvaluesMat);
    }

    Matrix eigenvector() // Calculate eigenvalue (only real eigenvalue)
    {
        if (row != column)
        {
            cerr << "\033[31;1mMatrix must be a square matrix.\033[0m" << endl;
            return Matrix(this->row, 0);
        }
        Mat Mat_1 = toMat(CV_64F);
        Mat eigenvaluesMat;
        Mat eigenvectorsMat;
        eigenNonSymmetric(Mat_1, eigenvaluesMat, eigenvectorsMat);
        return Matrix(eigenvectorsMat);
    }

    T trace()
    {
        if (column != row)
        {
            cerr << "\033[31;1mMatrix should be a square matrix.\033[0m" << endl;
            return NAN;
        }
        T trace = 0;
        for (int i = 0; i < column; i++)
        {
            trace += matrix[i][i];
        }
        return trace;
    }

    //使用伴随矩阵计算矩阵的逆
    //首先计算矩阵的行列式
    //而计算行列式需要计算代数余子式
    //计算代数余子式又需要行列式，所以计算行列式是一个递归过程
    Matrix smallMatrix(int m, int n) // 计算方块matrix去点m行和n列的矩阵
    {
        if (column != row)
        {
            cerr << "\033[31;1mMatrix should be a square matrix.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Matrix smallMatrix(row - 1, row - 1);
        for (int i = 0; i < row - 1; i++)
        {
            for (int j = 0; j < row - 1; j++)
            {
                if (i < m)
                {
                    if (j < n)
                    {
                        smallMatrix[i][j] = matrix[i][j];
                    }
                    else //第n列被抛弃
                    {
                        smallMatrix[i][j] = matrix[i][j + 1];
                    }
                }
                else //第m行被抛弃
                {
                    if (j < n)
                    {
                        smallMatrix[i][j] = matrix[i + 1][j];
                    }
                    else
                    {
                        smallMatrix[i][j] = matrix[i + 1][j + 1];
                    }
                }
            }
        }
        return smallMatrix;
    }

    double det() //计算行列式，注意只计算实数仿真的行列式，这点需要探讨
    {
        double det = 0;
        if (row != column)
        {
            cout << "\033[31;1mMatrix should be a square matrix.\033[0m" << endl;
            return NAN;
        }
        if (column == 1) //递归结束条件
        {
            return matrix[0][0];
        }
        else
        {

            for (int i = 0; i < column; i++) //使用矩阵的第一行计算行列式,一行的每一个元素都要递归一次
            {
                Matrix small_Matrix = smallMatrix(0, i); //奇怪 small_ 和 small
                det += matrix[0][i] * pow(-1, i) * small_Matrix.det();
            }
        }
        return det;
    }

    Matrix inverse()
    {
        if (column != row)
        {
            cerr << "\033[31;1mMatrix is not a square matrix.\033[0m" << endl;
            return Matrix(0, 0);
        }
        double det = this->det();
        if (det == 0)
        {
            cerr << "\033[31;1mMatrix is irreversible, determinant is zero.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Matrix matrix_inverse(column, column);
        for (int i = 0; i < column; i++)
        {
            for (int j = 0; j < column; j++)
            {
                matrix_inverse[j][i] = pow(-1, i + j) * smallMatrix(i, j).det() / det; //计算代数余子式
            }
        }
        return matrix_inverse;
    }

    //矩阵旋转180度
    Matrix rotate_180()
    {
        Matrix ans = Matrix(row, column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                ans[i][j] = matrix[row - 1 - i][column - 1 - j];
            }
        }
        return ans;
    }

    //矩阵卷积,mode = 0: same mode; mode=1: full mode; mode=2: valid mode.
    Matrix conv(Matrix kernel, int mode)
    {
        if (this->row != this->column)
        {
            cerr << "\033[31;1mMatrix is not a square matrix.\033[0m" << endl;
            return Matrix(0, 0);
        }
        if (kernel.row != kernel.column)
        {
            cerr << "\033[31;1mKernel is not a square matrix.\033[0m" << endl;
            return Matrix(0, 0);
        }
        if (kernel.column % 2 != 1)
        {
            cerr << "\033[31;1mKernel size must be odd.\033[0m" << endl;
            return Matrix(0, 0);
        }
        if (mode != 0 && mode != 1 && mode != 2)
        {
            cerr << "\033[31;1mmode is not 0, 1 or 2.\033[0m" << endl;
            return Matrix(0, 0);
        }

        int bias_row = kernel.row / 2;
        int bias_column = kernel.column / 2;

        if (mode == 1)
        {
            // cout <<"mode1";
            Matrix temp = Matrix(row + bias_row * 2, column + bias_column * 2);
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    temp[i + bias_row][j + bias_column] = matrix[i][j];
                }
            }
            return temp.conv(kernel, 0);
        }
        else if (mode == 2)
        {
            // cout <<"mode2";
            Matrix ans = Matrix(row - bias_row * 2, column - bias_column * 2);
            kernel = kernel.rotate_180();
            for (int i = bias_row; i < row - bias_row; i++)
            {
                for (int j = bias_column; j < column - bias_column; j++)
                {
                    for (int m = 0; m < kernel.row; m++)
                    {
                        for (int n = 0; n < kernel.column; n++)
                        {
                            // p , q是被卷积的坐标
                            int p = i + m - bias_row;
                            int q = j + n - bias_column;
                            ans[i - bias_row][j - bias_column] = ans[i - bias_row][j - bias_column] + matrix[p][q] * kernel[m][n];
                        }
                    }
                }
            }
            return ans;
        }
        else // mode = 0
        {
            kernel = kernel.rotate_180();
            Matrix ans = Matrix(row, column);
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    for (int m = 0; m < kernel.row; m++)
                    {
                        for (int n = 0; n < kernel.column; n++)
                        {
                            // p , q是被卷积的坐标
                            int p = i + m - bias_row;
                            int q = j + n - bias_column;
                            //默认padding是补0，所以直接忽略
                            if (p >= 0 && p < row && q >= 0 && q < column)
                            {
                                ans[i][j] = ans[i][j] + matrix[p][q] * kernel[m][n];
                            }
                        }
                    }
                }
            }
            return ans;
        }
    }
};
