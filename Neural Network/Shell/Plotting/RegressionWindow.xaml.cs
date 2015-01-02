using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using SharpNN.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Shell.Plotting
{
    /// <summary>
    /// Interaction logic for RegressionWindow.xaml
    /// </summary>
    public partial class RegressionWindow : Window
    {
        public RegressionWindow(PlotModel plotModel)
        {
            InitializeComponent();
            RegressionPlot.Model = plotModel;
        }
    }
}
