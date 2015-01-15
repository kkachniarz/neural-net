using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN.ActivationFunctions
{
    [Serializable]
    public class UnipolarSigmoidActivation : IActivation
    {
        public double MaxValue { get { return 1; } }
        public double MinValue { get { return 0; } }
        public double Calc(double x)
        {
            return (1.0 / (1.0 + Math.Pow(Math.E, -x)));
        }

        public double CalcDerivative(double x)
        {
            return x * (1.0 -x);
        }

        public string Name
        {
            get { return "Unipolar sigmoid"; }
        }


        public IActivation Clone()
        {
            return new UnipolarSigmoidActivation();
        }
    }
}
