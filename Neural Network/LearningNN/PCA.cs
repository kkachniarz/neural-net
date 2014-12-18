using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Windows.Documents;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace LearningNN
{
    public static class PCA
    {
        public static double EPSILON = 0.001d;
        public static void Run(IDataSet set, int destinationInputLenght)
        {
            // Preprocessing
            if (set == null
                || set.EnumeratePatterns().First() == null
                || set.EnumeratePatterns().First().Input == null)
                throw new ArgumentException();

            int inputLenght = set.EnumeratePatterns().First().Input.Count;

            if (destinationInputLenght >= inputLenght)
                return;

            // Input normalization
            double max = set.EnumeratePatterns()
                .Select(x => Math.Abs(x.Input.AbsoluteMaximum()))
                .ToList()
                .Max(x => x);

            foreach (var pattern in set.EnumeratePatterns())
            {
                pattern.Input = pattern.Input.Divide(max);
            }

            // Calculating matrix R
            Matrix<double> R = new DenseMatrix(inputLenght, inputLenght);
            foreach (var pattern in set.EnumeratePatterns())
            {
                R += pattern.Input.OuterProduct(pattern.Input);
            }
            R /= set.PatternCount;

            // QR decomposition
            var qr = R.QR(QRMethod.Full);

            // Calculating eigenvectors and eigenvalues
            var eigens = new List<Eigen>();
            var diagonal = qr.Q.Inverse() * R * qr.Q;
            for (int i = 0; i < inputLenght; i++)
            {
                eigens.Add(new Eigen
                {
                    Vector = qr.Q.Column(i),
                    Value = diagonal[i, i],
                });
            }
            
            //TestEigenValues(eigens, R, inputLenght);

            // Selecting eigenvectors with the biggest eigenvalues
            var pcaColumns = eigens
                    .OrderByDescending(x => x.Value)
                    .Take(destinationInputLenght)
                    .Select(x => x.Vector)
                    .ToList();

            // Creating transform matrix
            Matrix<double> pcaTransform  = new DenseMatrix(inputLenght, destinationInputLenght);
            for (int i = 0; i < destinationInputLenght; i++)
            {
                pcaTransform.SetColumn(i, pcaColumns[i]);
            }

            // Transforming input vectors
            foreach (var pattern in set.EnumeratePatterns())
            {
                pattern.Input = pattern.Input * pcaTransform;
            }
        }

        private static bool TestEigenValues(List<Eigen> eigens, Matrix<double> R, int inputLenght)
        {
            for (int i = 0; i < inputLenght; i++)
            {
                var eigensTestValues = eigens[i].Vector * R / eigens[i].Vector;
                if (eigensTestValues.ToList().Exists(x => Math.Abs(x - eigens[i].Value) > EPSILON))
                {
                    return false;
                }
            }

            return true;
        }

        public class Eigen
        {
            public Vector<double> Vector { get; set; }
            public double Value { get; set; }
        }
    }
}
