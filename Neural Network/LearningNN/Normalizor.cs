using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public class Normalizor
    {
        private double maxValueFrom;
        private double minValueFrom;
        private double maxValueTo; 
        private double minValueTo;
        public Normalizor(double _maxValueFrom, double _minValueFrom, double _maxValueTo, double _minValueTo)
	    {
            maxValueFrom = _maxValueFrom;
            minValueFrom = _minValueFrom;
            maxValueTo = _maxValueTo;
            minValueTo = _minValueTo;
	    }

        public double Normalize(double value)
        {
            if(value < minValueFrom || value > maxValueFrom)
            {
                throw new ArgumentException("Value is out of the range.");
            }

            return minValueTo + (maxValueTo - minValueTo) * ((value - minValueFrom)/(maxValueFrom - minValueFrom));
        }

        public double NormalizeBack(double value)
        {
            if (value < minValueTo || value > maxValueTo)
            {
                throw new ArgumentException("Value is out of the range.");
            }

            return minValueFrom + (maxValueFrom - minValueFrom) * ((value - minValueTo) / (maxValueTo - minValueTo));
        }
    }
}
