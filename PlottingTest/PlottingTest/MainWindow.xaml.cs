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

namespace PlottingTest
{
    /// <summary>
    /// Interaction logic for StartWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        Random r = new Random();

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Btn_Network_Error_Click(object sender, RoutedEventArgs e)
        {
            List<ErrorPoint> input = GenerateNetworkErrorDataPoints(); // this will be filled in during network training
            Window networkErrorWindow = new NetworkErrorWindow(input);
            networkErrorWindow.Show();
        }

        private void Regression_Click(object sender, RoutedEventArgs e)
        {
            List<RegressionPoint> actual = GenerateActualRegression();
            List<RegressionPoint> ideal = GenerateIdealRegression();
            Window regressionWindow = new RegressionWindow(actual, ideal);
            regressionWindow.Show();
        }

        private void Classification_Click(object sender, RoutedEventArgs e)
        {
            List<ClassificationPoint> actual = GenerateActualClassification();
            List<ClassificationPoint> ideal = GenerateIdealClassification();
            Window classificationWindow = new ClassificationWindow(actual, ideal);
            classificationWindow.Show();
        }

        private List<ErrorPoint> GenerateNetworkErrorDataPoints()
        {
            List<ErrorPoint> dataPoints = new List<ErrorPoint>();
            for (int i = 0; i < 10000; i++)
            {
                dataPoints.Add(new ErrorPoint(i, 1.0 / (double)(i + 1.0)));
            }

            return dataPoints;
        }

        private List<RegressionPoint> GenerateIdealRegression()
        {
            List<RegressionPoint> dataPoints = new List<RegressionPoint>();
            for (double i = -1.0; i < 1.0; i += 0.01)
            {
                dataPoints.Add(new RegressionPoint(i, i*i - 0.5));
            }

            return dataPoints;
        }

        private List<RegressionPoint> GenerateActualRegression()
        {
            List<RegressionPoint> dataPoints = new List<RegressionPoint>();
            for (double i = -1.0; i < 1.0; i += 0.1)
            {
                dataPoints.Add(new RegressionPoint(i, i * i - 0.4 + r.NextDouble()*0.1));
            }

            return dataPoints;
        }

        private List<ClassificationPoint> GenerateIdealClassification()
        {
            List<ClassificationPoint> dataPoints = new List<ClassificationPoint>();
            for (double x = -3.0; x < 3.0; x += 0.1)
            {
                for(double y = -3.0; y < 3.0; y += 0.1)
                {
                    dataPoints.Add(new ClassificationPoint(x, y, GetIdealClassID(x, y)));
                }
            }

            return dataPoints;
        }

        private List<ClassificationPoint> GenerateActualClassification()
        {
            List<ClassificationPoint> dataPoints = new List<ClassificationPoint>();
            for (double x = -3.0; x < 3.0; x += 0.1)
            {
                for(double y = -3.0; y < 3.0; y += 0.1)
                {
                    dataPoints.Add(new ClassificationPoint(x + (r.NextDouble()-0.5), y + (r.NextDouble()-0.5), GetIdealClassID(x, y)));
                }
            }

            return dataPoints;
        }

        private int GetIdealClassID(double x, double y)
        {
            if(x < 0 && y < 0)
            {
                return 0;
            }
            if(x >= 0 && y < 0)
            {
                return 1;
            }
            if(x < 0 && y >=0)
            {
                return 2;
            }

            return 3;
        }
    }
}
