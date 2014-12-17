using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN
{
    public class Pattern
    {
        public Vector<double> Input { get; set; }
        public Vector<double> IdealOutput { get; set; }
        public Vector<double> NetworkAnswer { get; set; }
        public int TimeIndex { get; private set; }

        public Pattern(int timeIndex)
        {
            this.Input = null;
            this.IdealOutput = null;
            this.NetworkAnswer = null;
            this.TimeIndex = timeIndex;
        }

        public Pattern(int timeIndex, Vector<double> input)
            : this(timeIndex)
        {
            this.Input = input;
        }
    }
}
