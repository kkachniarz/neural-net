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
        const double MARGIN_FACTOR = 0.2;

        private double maxValueFrom;
        private double minValueFrom;
        private double maxValueTo; 
        private double minValueTo;
        private double minValueToCorrected;
        private double maxValueToCorrected;

        public Normalizor(double _minValueFrom, double _maxValueFrom, double _minValueTo, double _maxValueTo)
	    {
            maxValueFrom = _maxValueFrom;
            minValueFrom = _minValueFrom;
            maxValueTo = _maxValueTo;
            minValueTo = _minValueTo;
            double margin = (maxValueTo - minValueTo) * MARGIN_FACTOR;
            minValueToCorrected = minValueTo + margin;
            maxValueToCorrected = maxValueTo - margin;
	    }

        public double Normalize(double value)
        {
            ValidateWithinBoundsEps(value, minValueFrom, maxValueFrom);
            return minValueToCorrected + (maxValueToCorrected - minValueToCorrected) * ((value - minValueFrom) / (maxValueFrom - minValueFrom));
        }

        public double NormalizeBack(double value)
        {
            ValidateWithinBoundsEps(value, minValueTo, maxValueTo);
            return minValueFrom + (maxValueFrom - minValueFrom) * ((value - minValueToCorrected) / (maxValueToCorrected - minValueToCorrected));
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
