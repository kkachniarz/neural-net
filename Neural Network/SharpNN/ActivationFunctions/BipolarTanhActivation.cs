using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN.ActivationFunctions
{
    public class BipolarTanhActivation : IActivation
    {
        public double MaxValue { get { return 1; } }
        public double MinValue { get { return -1; } }

        public double Calc(double x)
        {
            return Math.Tanh(x);
        }

        public double CalcDerivative(double x)
        {
            return 1.0 - Math.Pow(Calc(x), 2.0);
        }

        public string Name
        {
            get { return "Bipolar Tanh"; }
        }

        public IActivation Clone()
        {
            return new BipolarTanhActivation();
        }
    }
}
