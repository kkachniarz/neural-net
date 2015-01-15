
using System;
namespace SharpNN.ActivationFunctions
{
    [Serializable]
    public class LinearActivation : IActivation
    {
        public double MaxValue { get { return double.MaxValue; } }
        public double MinValue { get { return double.MinValue; } }
        public double Calc(double x)
        {
            return x;
        }

        public double CalcDerivative(double x)
        {
            return 1;
        }

        public string Name
        {
            get { return "Linear"; }
        }

        public IActivation Clone()
        {
            return new LinearActivation();
        }
    }
}
