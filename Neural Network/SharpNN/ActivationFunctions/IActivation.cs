using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNN
{
    public interface IActivation
    {
        double MaxValue { get; }
        double MinValue { get; }
        double Calc(double x);
        double CalcDerivative(double x);

        string Name { get; }
    }
}
