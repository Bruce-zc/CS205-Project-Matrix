#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "Matrix.h"

using namespace std;

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
            answer += matrix[i][0] * other[0][i];
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
};
