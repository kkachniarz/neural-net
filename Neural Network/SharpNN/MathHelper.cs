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
        private const double EPSILON_FACTOR = 0.2;
        public static double RandomExceptZero(double maxAbsValue, double epsilonFactor = EPSILON_FACTOR)
        {
            double eps = epsilonFactor * maxAbsValue;
            AssertCanRandomize(maxAbsValue, eps);
            double number = (Math.Abs(Rand.NextDouble()) * (maxAbsValue - eps) + eps);
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

            if(maxAbsValue * 0.8 <=  epsilon)
            {
                throw new ArgumentException("Maximum absolute value must be greater than 0.8 * epsilon value");
            }
        }
    }
}
