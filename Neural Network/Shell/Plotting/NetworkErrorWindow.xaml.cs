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
    /// Interaction logic for NetworkErrorWindow.xaml
    /// </summary>
    public partial class NetworkErrorWindow : Window
    {
        private const double SCALE = 1000.0;
        private PlotModel plotModel;

        public NetworkErrorWindow(List<double> trainingSetErrors, List<double> validationSetErorrs = null)
        {
            InitializeComponent();
            plotModel = new PlotModel();
            SetUpModel(trainingSetErrors, validationSetErorrs);
            ErrorPlot.Model = plotModel;
        }

        private void SetUpModel(List<double> trainingSetErrors, List<double> validationSetErrors = null)
        {
            var linearAxis1 = new LinearAxis();
            linearAxis1.TickStyle = TickStyle.Outside;
            linearAxis1.Position = AxisPosition.Left;
            linearAxis1.Minimum = 0;
            linearAxis1.Maximum = trainingSetErrors.Max() * 1.1 * SCALE;
            linearAxis1.Title = "Error";
            plotModel.Axes.Add(linearAxis1);
            var linearAxis2 = new LinearAxis();
            linearAxis2.Position = AxisPosition.Bottom;
            linearAxis2.TickStyle = TickStyle.Outside;
            linearAxis2.Minimum = 0;
            linearAxis2.Maximum = trainingSetErrors.Count;
            linearAxis2.Title = "Iteration";
            plotModel.Axes.Add(linearAxis2);

            var series1 = new LineSeries { Title = "Training set error", MarkerType = MarkerType.None };

            series1.Points.AddRange(CreateDataPoints(trainingSetErrors));
            plotModel.Series.Add(series1);
            if (validationSetErrors != null)
            {
                var series2 = new LineSeries { Title = "Validation set error", MarkerType = MarkerType.None };
                series2.Points.AddRange(CreateDataPoints(validationSetErrors));
                plotModel.Series.Add(series2);
            }
        }

        private List<DataPoint> CreateDataPoints(List<double> mseHistory)
        {
            List<DataPoint> points = new List<DataPoint>();
            for (int i = 0; i < mseHistory.Count; i++)
            {
                points.Add(new DataPoint(i, mseHistory[i] * SCALE));
            }
            return points;
        }
    }
}
