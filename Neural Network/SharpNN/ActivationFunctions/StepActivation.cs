
namespace SharpNN.ActivationFunctions
{
    public class StepActivation : IActivation
    {
        public double MaxValue { get { return 1; } }
        public double MinValue { get { return 0; } }
        public double Calc(double x)
        {
            return x < 0 ? 0.0 : 1.0;
        }

        public double CalcDerivative(double x)
        {
            return 1;
        }
    }
}
