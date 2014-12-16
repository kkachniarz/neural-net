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

namespace Neural_Network.Plotting
{
    /// <summary>
    /// Interaction logic for RegressionWindow.xaml
    /// </summary>
    public partial class RegressionWindow : Window
    {
        private PlotModel plotModel;

        public RegressionWindow(List<RegressionPoint> trainingPoints, List<RegressionPoint> idealAnswer, List<RegressionPoint> networkAnswer = null)
        {
            InitializeComponent();
            plotModel = new PlotModel();
            if(networkAnswer == null)
            {
                networkAnswer = new List<RegressionPoint>();
            }

            SetUpModel(trainingPoints, idealAnswer, networkAnswer);
            RegressionPlot.Model = plotModel;
        }

        private void SetUpModel(List<RegressionPoint> trainingPoints, List<RegressionPoint> idealAnswer, List<RegressionPoint> networkAnswer)
        {
            var series1 = new ScatterSeries { Title = "Training data", MarkerType = MarkerType.Triangle, MarkerSize = 2.0, MarkerFill = OxyColors.DarkGray };
            trainingPoints.ForEach(x => series1.Points.Add(CreateScatterPoint(x)));
            var series2 = new ScatterSeries { Title = "Ideal answer", MarkerType = MarkerType.Circle, MarkerSize = 2.0, MarkerFill = OxyColors.Black };
            idealAnswer.ForEach(x => series2.Points.Add(CreateScatterPoint(x)));
            var series3 = new ScatterSeries { Title = "Network answer", MarkerType = MarkerType.Diamond, MarkerSize = 4.0, MarkerFill = OxyColors.Red };
            networkAnswer.ForEach(x => series3.Points.Add(CreateScatterPoint(x)));

            double minX = Math.Min(trainingPoints.Min(p => p.X), networkAnswer.Concat(idealAnswer).Min(p => p.X));
            double maxX = Math.Max(trainingPoints.Max(p => p.X), networkAnswer.Concat(idealAnswer).Max(p => p.X));
            double minY = Math.Min(trainingPoints.Min(p => p.Y), networkAnswer.Concat(idealAnswer).Min(p => p.Y));
            double maxY = Math.Max(trainingPoints.Max(p => p.Y), networkAnswer.Concat(idealAnswer).Max(p => p.Y));

            var linearAxis1 = new LinearAxis();
            linearAxis1.TickStyle = TickStyle.Outside;
            linearAxis1.Position = AxisPosition.Left;
            linearAxis1.Minimum = minY;
            linearAxis1.Maximum = maxY;
            linearAxis1.Title = "Y";
            plotModel.Axes.Add(linearAxis1);

            var linearAxis2 = new LinearAxis();
            linearAxis2.Position = AxisPosition.Bottom;
            linearAxis2.TickStyle = TickStyle.Outside;
            linearAxis2.Minimum = minX;
            linearAxis2.Maximum = maxX;
            linearAxis2.Title = "X";
            plotModel.Axes.Add(linearAxis2);

            plotModel.Series.Add(series1);
            plotModel.Series.Add(series2);
            plotModel.Series.Add(series3);
        }

        private ScatterPoint CreateScatterPoint(RegressionPoint regressionDataPoint)
        {
            return new ScatterPoint(regressionDataPoint.X, regressionDataPoint.Y);
        }

        private DataPoint CreateDataPoint(RegressionPoint regressionDataPoint)
        {
            return new DataPoint(regressionDataPoint.X, regressionDataPoint.Y);
        }
    }
}
