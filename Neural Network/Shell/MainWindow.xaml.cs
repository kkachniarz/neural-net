using LearningNN;
using LearningNN.DataSet;
using LearningNN.LearningStrategy;
using MathNet.Numerics.LinearAlgebra;
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
        public bool IsReady { get { return csvLines != null; } }

        List<DenseVector> csvLines;
        IDataSet testDataSet;
        IDataSet trainDataSet;
        ILearningStrategy learningStrategy;
        string dataSetPath;

        public MainWindow()
        {
            InitializeComponent();

            NetworkTypeCombobox.ItemsSource = Enum.GetValues(typeof(NetworkType)).Cast<NetworkType>();
            ProblemTypeCombobox.ItemsSource = Enum.GetValues(typeof(PartIIProblemType)).Cast<PartIIProblemType>();
            ActivationCombobox.ItemsSource = Enum.GetValues(typeof(ActivationFunction)).Cast<ActivationFunction>();
            BiasCombobox.ItemsSource = Enum.GetValues(typeof(YesNo)).Cast<YesNo>();

            ActivationCombobox.SelectedIndex = 0;
            BiasCombobox.SelectedIndex = 0;
        }

        private void ReadDataSet(object sender, RoutedEventArgs e)
        {
            string shortName;
            dataSetPath = ReadCSVFile(out shortName);

            if (dataSetPath == null)
                return;

            csvLines = FileManager.ReadDataFromCSV(dataSetPath);
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
            int maxIterations = int.Parse(MaxIterations.Text);
            int badIterations = BadIterations.Text == ""? maxIterations : int.Parse(BadIterations.Text); // ignore or use
            double learningRate = LearningRateSlider.Value;
            double momentum = MomentumSlider.Value;
            float trainSetPercentage = float.Parse(TrainSetPercentage.Text, CultureInfo.InvariantCulture);
            int outputCount = int.Parse(OutputCount.Text);
            NetworkType networkType = (NetworkType)NetworkTypeCombobox.SelectedItem;
            int ctsPrevValuesCount = int.Parse(CTSPreviousValues.Text);

            PartIIProblemType problemType = (PartIIProblemType)ProblemTypeCombobox.SelectedItem;

            if (problemType == PartIIProblemType.CTS)
            {
                InitCTS(layersVal, trainSetPercentage, ctsPrevValuesCount);
            }
            else
            {
                InitStock(layersVal, trainSetPercentage, outputCount);
            }

            layersVal.Add(outputCount);
            INetwork network = null;
            switch(networkType)
            {
                case NetworkType.MLP:
                    network = new NeuralNetwork(activation, bias, layersVal.ToArray());
                    break;
                case NetworkType.Jordan:
                    network = new RecursiveNetwork(RecursiveNetwork.Type.Jordan,
                    activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                    break;
                case NetworkType.Elman:
                    network = new RecursiveNetwork(RecursiveNetwork.Type.Elman,
                    activation, bias, layersVal[0], layersVal[1], layersVal[2]);
                    break;
            }

            CheckIfPerformPCA();

            NormalizeData(network, trainDataSet, testDataSet);

            VSetLearningStrategy vSetStrategy = new VSetLearningStrategy(learningRate, momentum, 0.2f);
            vSetStrategy.IterLimit = maxIterations;
            vSetStrategy.MaxBadIterations = badIterations;
            learningStrategy = vSetStrategy;

            var learningResult = BackpropagationManager.Run(network, trainDataSet, testDataSet,
                learningStrategy);

            NormalizeDataBack(network, trainDataSet, testDataSet);

            ShowNetworkErrorWindow(learningResult);
            Show1DRegression(trainDataSet, testDataSet, false);
        }

        private void CheckIfPerformPCA()
        {
            int dimensionPca;

            if (int.TryParse(this.PCA.Text, out dimensionPca))
            {
                LearningNN.PCA.Run(trainDataSet, dimensionPca);
                LearningNN.PCA.Run(testDataSet, dimensionPca);
            }
        }

        private static void NormalizeData(INetwork network, IDataSet trainData, IDataSet testData)
        {
            var extremums = DataExtremumsForNetwork.Merge(trainData.Extremums, testData.Extremums);

            trainData.Normalize(network.Activation.MinValue, network.Activation.MaxValue, extremums);
            testData.Normalize(network.Activation.MinValue, network.Activation.MaxValue, extremums);
        }

        private static void NormalizeDataBack(INetwork network, IDataSet trainData, IDataSet testData)
        {
            testData.NormalizeBack();
            trainData.NormalizeBack();
        }

        private void InitCTS(List<int> layersVal, float trainSetPercentage, int historyLength)
        {
            int dimensionPca;
            if (int.TryParse(this.PCA.Text, out dimensionPca))
            {
                layersVal.Insert(0, Math.Min(dimensionPca, historyLength));
            }
            else
            {
                layersVal.Insert(0, historyLength); // network needs as many inputs as many historical values we feed it.
            }

            int trainSetEndIndex = (int)(trainSetPercentage * csvLines.Count);
            List<DenseVector> chaoticValues = csvLines; // no need for further parsing

            List<DenseVector> trainValues = chaoticValues.ExtractList(0, trainSetEndIndex);
            List<DenseVector> testValues = chaoticValues.ExtractList(trainSetEndIndex, chaoticValues.Count);

            trainDataSet = new ChaoticDataSet(trainValues, historyLength, 0);
            if (trainSetEndIndex >= chaoticValues.Count - 1)
            {
                testDataSet = trainDataSet.Clone();
            }
            else
            {
                testDataSet = new ChaoticDataSet(testValues, historyLength, trainSetEndIndex);
            }
        }

        private void InitStock(List<int> layersVal, float trainSetPercentage, int outputCount)
        {
            int dimensionPca;
            int dispalyInputCount = csvLines[0].Count - outputCount; // this way we can calculate number of display inputs

            if (int.TryParse(this.PCA.Text, out dimensionPca))
            {
                layersVal.Insert(0, Math.Min(dimensionPca, dispalyInputCount));
            }
            else
            {
                layersVal.Insert(0, dispalyInputCount);
            }
            
            
            int trainSetEndIndex = (int)(trainSetPercentage * csvLines.Count);

            List<DenseVector> allInputs = csvLines.Select(v => v.CreateSubVector(0, dispalyInputCount)).ToList();
            List<DenseVector> allOutputs = csvLines.Select(v => v.CreateSubVector(dispalyInputCount, outputCount)).ToList();

            trainDataSet = new StockDataSet(allInputs.ExtractList(0, trainSetEndIndex), 
                allOutputs.ExtractList(0, trainSetEndIndex), 0);

            if (trainSetEndIndex >= allInputs.Count - 1)
            {
                testDataSet = trainDataSet;
            }
            else
            {
                testDataSet = new StockDataSet(allInputs.ExtractList(trainSetEndIndex, csvLines.Count),
                    allOutputs.ExtractList(trainSetEndIndex, csvLines.Count), trainSetEndIndex);
            }
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

        private void Show1DRegression(IDataSet trainingSet, IDataSet testSet, bool debugPlot) // debugPlot doesn't use time index, just X -> Y mapping values
        {
            List<RegressionPoint> trainPoints = new List<RegressionPoint>();
            List<RegressionPoint> testIdealPoints = new List<RegressionPoint>();
            List<RegressionPoint> networkAnswers = new List<RegressionPoint>();
            Func<Pattern, double> patternToDouble;
            if(debugPlot)
            {
                patternToDouble = p => p.Input[0];
            }
            else
            {
               patternToDouble = p => p.TimeIndex;
            }

            foreach(Pattern p in trainingSet.EnumeratePatterns())
            {
                trainPoints.Add(new RegressionPoint(patternToDouble(p), p.IdealOutput.At(0)));
            }

            foreach(Pattern p in testSet.EnumeratePatterns())
            {
                testIdealPoints.Add(new RegressionPoint(patternToDouble(p), p.IdealOutput.At(0)));
                networkAnswers.Add(new RegressionPoint(patternToDouble(p), p.NetworkAnswer.At(0)));
            }

            Window regressionWindow = new RegressionWindow(
                trainPoints,
                testIdealPoints,
                networkAnswers);
            regressionWindow.Show();
        }
    }

    public enum YesNo
    {
        Yes,
        No
    }

    public enum ActivationFunction
    {
        Unipolar,
        Bipolar
    }

    public enum PartIIProblemType
    {
        CTS,
        Stock
    }
}
