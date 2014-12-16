
namespace SharpNN.ActivationFunctions
{
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
    }
}
