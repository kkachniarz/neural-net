using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN
{
    public static class MathHelper
    {
        public static Random Rand = new Random();
        private const double EPSILON = 0.00001;
        public static double RandomExceptZero(double maxAbsValue, double epsilon = EPSILON)
        {
            AssertCanRandomize(maxAbsValue, epsilon);
            double number = (Rand.NextDouble() * (maxAbsValue - epsilon) + epsilon);
            if(Rand.NextDouble() < 0.5)
            {
                number = -number;
            }

            return number;
        }

        private static void AssertCanRandomize(double maxAbsValue, double epsilon)
        {
            if(maxAbsValue <= 0)
            {
                throw new ArgumentException("Maximum absolute value must be greater than 0");
            }

            if(maxAbsValue <= epsilon)
            {
                throw new ArgumentException("Maximum absolute value must be greater than epsilon value");
            }
        }
    }
}
