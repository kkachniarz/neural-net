using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.DataSet
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

        public Pattern Clone()
        {
            Pattern ret = new Pattern(this.TimeIndex);
            if (this.Input != null)
            {
                ret.Input = (Vector<double>)this.Input.Clone();
            }
            if (this.IdealOutput != null)
            {
                ret.IdealOutput = (Vector<double>)this.IdealOutput.Clone();
            }
            if (this.NetworkAnswer != null)
            {
                ret.NetworkAnswer = (Vector<double>)this.NetworkAnswer.Clone();
            }

            return ret;
        }
    }
}
