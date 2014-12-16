using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public static class Extensions
    {
        public static DenseVector CreateSubVector(this DenseVector vector, int startIndex, int count)
        {
            DenseVector ret = new DenseVector(count);
            vector.CopySubVectorTo(ret, startIndex, 0, count);
            return ret;
        }
    }
}
