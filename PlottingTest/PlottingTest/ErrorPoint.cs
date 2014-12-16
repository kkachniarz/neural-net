using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PlottingTest
{
    public class ErrorPoint
    {
        public double Iteration { get; private set; }
        public double Error { get; private set; }

        public ErrorPoint(double iteration, double error)
        {
            this.Iteration = iteration;
            this.Error = error;
        }
    }
}
