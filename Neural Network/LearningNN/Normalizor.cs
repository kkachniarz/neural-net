using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public class Normalizor
    {
        const double EPSILON = 1E-5;
        private double maxValueFrom;
        private double minValueFrom;
        private double maxValueTo; 
        private double minValueTo;
        public Normalizor(double _minValueFrom, double _maxValueFrom, double _minValueTo, double _maxValueTo)
	    {
            maxValueFrom = _maxValueFrom;
            minValueFrom = _minValueFrom;
            maxValueTo = _maxValueTo;
            minValueTo = _minValueTo;
	    }

        public double Normalize(double value)
        {
            ValidateWithinBoundsEps(value, minValueFrom, maxValueFrom);
            return minValueTo + (maxValueTo - minValueTo) * ((value - minValueFrom)/(maxValueFrom - minValueFrom));
        }

        public double NormalizeBack(double value)
        {
            ValidateWithinBoundsEps(value, minValueTo, maxValueTo);
            return minValueFrom + (maxValueFrom - minValueFrom) * ((value - minValueTo) / (maxValueTo - minValueTo));
        }

        private void ValidateWithinBoundsEps(double value, double min, double max)
        {
            if (value < min - EPSILON || value > max + EPSILON)
            {
                throw new ArgumentException("Value is out of the range.");
            }
        }
    }
}
