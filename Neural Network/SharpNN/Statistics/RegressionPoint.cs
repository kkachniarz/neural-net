using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN.Statistics
{
    public class RegressionPoint
    {
        public double X { get; private set; }
        public double Y { get; private set; }

        public RegressionPoint(double x, double y)
        {
            this.X = x;
            this.Y = y;
        }
    }
}
