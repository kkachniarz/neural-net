using LearningNN;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.Win32;
using Neural_Network;
using Neural_Network.Plotting;
using RecursiveNN;
using SharpNN;
using SharpNN.ActivationFunctions;
using SharpNN.Statistics;
using Shell.Enums;
using System;
using System.Collections.Generic;
using System.Globalization;
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

namespace Neural_Network
{
    public partial class MainWindow : Window
    {
        public bool IsReady { get { return dataSet != null; } }

        List<DenseVector> dataSet;
        string dataSetPath;

        public MainWindow()
        {
            InitializeComponent();

            NetworkTypeCombobox.ItemsSource = Enum.GetValues(typeof(NetworkType)).Cast<NetworkType>();
            ActivationCombobox.ItemsSource = Enum.GetValues(typeof(ActivationFunction)).Cast<ActivationFunction>();
            BiasCombobox.ItemsSource = Enum.GetValues(typeof(YesNo)).Cast<YesNo>();
            AppendTestCombobox.ItemsSource = Enum.GetValues(typeof(YesNo)).Cast<YesNo>();

            ActivationCombobox.SelectedIndex = 1;
            BiasCombobox.SelectedIndex = 0;
            AppendTestCombobox.SelectedIndex = 1;
        }

        private void ReadDataSet(object sender, RoutedEventArgs e)
        {
            string shortName;
            dataSetPath = ReadCSVFile(out shortName);

            if (dataSetPath == null)
                return;

            dataSet = FileManager.ReadDataFromCSV(dataSetPath);
            TrainSetLabel.Content = shortName;

            if (IsReady)
            {
                StartButton.IsEnabled = true;
            }
        }

        private string ReadCSVFile(out string shortName)
        {
            OpenFileDialog csvDlg = new OpenFileDialog();
            csvDlg.DefaultExt = ".csv";
            csvDlg.Filter = "CSV documents (.csv)|*.csv";
            csvDlg.Title = "Select proper CSV file";

            var result = csvDlg.ShowDialog();

            if (result == true)
            {
                var fileName = csvDlg.FileName.ToString();
                shortName = csvDlg.SafeFileName;
                return fileName;
            }

            shortName = null;
            return null;
        }

        private void StartButtonClick(object sender, RoutedEventArgs e)
        {
            var layers = LayersTextBox.Text.Split(new string[] { ",", " ", "-", "_", "." }, StringSplitOptions.RemoveEmptyEntries);
            List<int> layersVal = new List<int>();

            foreach (var layer in layers)
            {
                layersVal.Add(int.Parse(layer));
            }

            bool bias = (YesNo)BiasCombobox.SelectedItem == YesNo.Yes;
            IActivation activation = ((ActivationFunction)ActivationCombobox.SelectedItem == ActivationFunction.Bipolar) ? 
                (IActivation)new BipolarTanhActivation() : new UnipolarSigmoidActivation();
            int iterations = int.Parse(Iterations.Text);
            double learningRate = LearningRateSlider.Value;
            double momentum = MomentumSlider.Value;
            float trainSetPercentage = float.Parse(TrainSetPercentage.Text, CultureInfo.InvariantCulture);
            int outputCount = int.Parse(OutputCount.Text);
            NetworkType networkType = (NetworkType)NetworkTypeCombobox.SelectedItem;
            int historyLength = networkType == NetworkType.MLP? int.Parse(MLPHistoryLength.Text) : 0;
            BackpropagationRunMode runMode = BackpropagationRunMode.Online; // visit training examples in order provided
            // it's always regression: (x1, x2, ..., xn) -> (y1, y2, ..., ym) real values
            CasesData trainingCases;
            CasesData testCases;
            YesNo appendTestFile = (YesNo)AppendTestCombobox.SelectedItem;

            int input = dataSet[0].Count() - outputCount;
            CasesData.InitializeForPrediction(dataSet, out trainingCases, out testCases, outputCount, historyLength, trainSetPercentage);
            layersVal.Insert(0, input * (historyLength + 1));
            layersVal.Add(outputCount);

            INetwork network = null;
            switch(networkType)
            {
                case NetworkType.MLP:
                    network = new NeuralNetwork(activation, bias, layersVal.ToArray());
                    break;
                case NetworkType.Jordan:
                    network = new RecursiveNetwork(RecursiveNN.RecursiveNetwork.Type.Jordan,
                    activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                    break;
                case NetworkType.Elman:
                    network = new RecursiveNetwork(RecursiveNN.RecursiveNetwork.Type.Elman,
                    activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                    break;
            }


            LearningResult learningResult = BackpropagationManager.Run(network, trainingCases, testCases,
                runMode, iterations, learningRate, momentum);

            if (appendTestFile == YesNo.Yes)
            {
                AppendCSVile(dataSetPath, testCases);
            }

            ShowNetworkErrorWindow(learningResult);
            ShowRegressionWindow(trainingCases, testCases);
        }

        private void AppendCSVile(string path, CasesData data)
        {
            var output = new List<DenseVector>();

            for (int i = 0; i < data.CasesCount; i++)
            {
                output.Add(
                    data.ProblemType == ProblemType.Classification ?
                    data.GetClasificationOutput(i) :
                    data.GetIdealOutput(i));
            }

            FileManager.AppendDataToCSV(path, output);
        }

        private void ShowNetworkErrorWindow(LearningResult result)
        {
            Window errorWindow = new NetworkErrorWindow(result.MSEHistory);
            errorWindow.Show();
        }

        private void ShowRegressionWindow(CasesData trainingSet, CasesData testSet)
        {
            Window regressionWindow = new RegressionWindow(
                MakeRegressionPointList(trainingSet, trainingSet.GetIdealOutput), 
                MakeRegressionPointList(testSet, testSet.GetIdealOutput),
                MakeRegressionPointList(testSet, testSet.GetNetworkAnswer)); 
            regressionWindow.Show();
        }

        private List<RegressionPoint> MakeRegressionPointList(CasesData dataSet, Func<int, DenseVector> outputGetter)
        {
            List<RegressionPoint> points = new List<RegressionPoint>();
            for (int i = 0; i < dataSet.CasesCount; i++)
            {
                points.Add(new RegressionPoint(dataSet.GetInput(i).At(0), outputGetter(i).At(0)));
            }

            return points;
        }

        private void ShowClassificationWindow(CasesData trainingSet, CasesData testSet)
        {
            List<ClassificationPoint> testPoints = new List<ClassificationPoint>();
            testPoints.AddRange(MakeClassificationPointList(testSet, testSet.GetClasificationOutput));

            List<ClassificationPoint> idealPoints = new List<ClassificationPoint>();
            idealPoints.AddRange(MakeClassificationPointList(trainingSet, trainingSet.GetClasificationOutput));

            Window classificationWindow = new ClassificationWindow(testPoints, idealPoints, trainingSet.ClassIndexes);
            classificationWindow.Show();
        }

        private IList<ClassificationPoint> MakeClassificationPointList(CasesData dataSet, Func<int, DenseVector> outputGetter)
        {
            IList<ClassificationPoint> points = new List<ClassificationPoint>();
            for (int i = 0; i < dataSet.CasesCount; i++)
            {
                points.Add(new ClassificationPoint(dataSet.GetInput(i).At(0),
                    dataSet.GetInput(i).At(1), (int)outputGetter(i).At(0)));
            }

            return points;
        }
    }
}
