using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using SharpNN.Statistics;
using Shell.Plotting;
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

namespace Neural_Network.Plotting
{
    /// <summary>
    /// Interaction logic for NetworkErrorWindow.xaml
    /// </summary>
    public partial class NetworkErrorWindow : Window
    {
        public NetworkErrorWindow(PlotModel plotModel, string identifier)
        {
            InitializeComponent();
            ErrorPlot.Model = plotModel;
            Title = string.Format("{0}", identifier);
        }
    }
}
