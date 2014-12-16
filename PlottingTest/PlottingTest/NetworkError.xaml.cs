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
using System.Windows.Navigation;
using System.Windows.Shapes;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace PlottingTest
{
    public partial class NetworkErrorWindow : Window
    {
        private PlotModel plotModel;

        public NetworkErrorWindow(List<ErrorPoint> data, List<ErrorPoint> validationSetErorrs = null)
        {
            InitializeComponent();
            plotModel = new PlotModel();
            SetUpModel(data, validationSetErorrs);
            ErrorPlot.Model = plotModel;
        }

        private void SetUpModel(List<ErrorPoint> errorDataPoints, List<ErrorPoint> validationSetErrors = null)
        {
            var linearAxis1 = new LinearAxis();
            linearAxis1.TickStyle = TickStyle.Outside;
            linearAxis1.Position = AxisPosition.Left;
            linearAxis1.Minimum = 0;
            linearAxis1.Maximum = errorDataPoints.Count > 1000? errorDataPoints[20].Error : 0.5;
            linearAxis1.Title = "Error";
            plotModel.Axes.Add(linearAxis1);
            var linearAxis2 = new LinearAxis();
            linearAxis2.Position = AxisPosition.Bottom;
            linearAxis2.TickStyle = TickStyle.Outside;
            linearAxis2.Minimum = 0;
            linearAxis2.Maximum = errorDataPoints.Count;
            linearAxis2.Title = "Iteration";
            plotModel.Axes.Add(linearAxis2);

            var series1 = new LineSeries { Title = "Training set error", MarkerType = MarkerType.None };
            errorDataPoints.ForEach(x => series1.Points.Add(CreateDataPoint(x)));
            plotModel.Series.Add(series1);
            if(validationSetErrors != null)
            {
                var series2 = new LineSeries { Title = "Validation set error", MarkerType = MarkerType.None };
                errorDataPoints.ForEach(x => series2.Points.Add(CreateDataPoint(x)));
                plotModel.Series.Add(series2);
            }
        }

        private DataPoint CreateDataPoint(ErrorPoint errorDataPoint)
        {
            return new DataPoint(errorDataPoint.Iteration, errorDataPoint.Error);
        }
    }
}
