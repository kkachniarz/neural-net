using LearningNN;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.Win32;
using Neural_Network.Plotting;
using SharpNN;
using SharpNN.ActivationFunctions;
using SharpNN.Statistics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
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
using System.Windows.Threading;
using System.Xml;

namespace Neural_Network
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class LegacyMainWindow : Window
    {
        public bool IsReady { get { return trainingData != null && testData != null; } }
        
        List<DenseVector> trainingData;
        List<DenseVector> testData;
        string testDataPath;
        string trainDataPath;

        public LegacyMainWindow()
        {
            InitializeComponent();

            ActivationCombobox.ItemsSource = Enum.GetValues(typeof(ActivationFunction)).Cast<ActivationFunction>();
            BiasCombobox.ItemsSource = Enum.GetValues(typeof(YesNo)).Cast<YesNo>();
            RunCombobox.ItemsSource = Enum.GetValues(typeof(BackpropagationRunMode)).Cast<BackpropagationRunMode>();
            ProblemCombobox.ItemsSource = Enum.GetValues(typeof(ProblemType)).Cast<ProblemType>();
            AppendTestCombobox.ItemsSource = Enum.GetValues(typeof(YesNo)).Cast<YesNo>();

            ActivationCombobox.SelectedIndex = 1;
            BiasCombobox.SelectedIndex = 0;
            RunCombobox.SelectedIndex = 1;
            ProblemCombobox.SelectedIndex = 0;
            AppendTestCombobox.SelectedIndex = 1;
        }

        private void ReadTrainSet(object sender, RoutedEventArgs e)
        {
            string shortName;
            trainDataPath = ReadCSVFile(out shortName);

            if (trainDataPath == null)
                return;

            trainingData = FileManager.ReadDataFromCSV(trainDataPath);
            TrainSetLabel.Content = shortName;

            if(IsReady)
            {
                StartButton.IsEnabled = true;
            }
        }

        private void ReadTestSet(object sender, RoutedEventArgs e)
        {
            string shortName;
            testDataPath = ReadCSVFile(out shortName);

            if (testDataPath == null)
                return;

            testData = FileManager.ReadDataFromCSV(testDataPath);
            TestSetLabel.Content = shortName;

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

            if(result == true)
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

            foreach(var layer in layers)
            {
                layersVal.Add(int.Parse(layer));
            }

            YesNo bias = (YesNo)BiasCombobox.SelectedItem;
            ActivationFunction activation = (ActivationFunction)ActivationCombobox.SelectedItem;
            int iterations = int.Parse(Iterations.Text);
            double learningRate = LearningRateSlider.Value;
            double momentum = MomentumSlider.Value;
            BackpropagationRunMode runMode = (BackpropagationRunMode)RunCombobox.SelectedItem;
            ProblemType problem = (ProblemType)ProblemCombobox.SelectedItem;
            CasesData trainingCases;
            CasesData testCases;
            YesNo appendTestFile = (YesNo)AppendTestCombobox.SelectedItem;

            int input = testData.First().Count();
            int output = trainingData.First().Count() - input;

            CasesData.InitializeAllData(trainingData, testData, problem, out trainingCases, out testCases);

            layersVal.Insert(0, input);
            layersVal.Add((problem == ProblemType.Regression) ? output : trainingCases.ClassCount);

            NeuralNetwork network = new NeuralNetwork(
            (activation == ActivationFunction.Bipolar) ? (IActivation)new BipolarTanhActivation() : new UnipolarSigmoidActivation(),
            bias == YesNo.Yes, layersVal.ToArray());

            LearningResult learningResult = BackpropagationManager.Run(network, trainingCases, testCases, 
                runMode, iterations, learningRate, momentum);

            if (appendTestFile == YesNo.Yes)
            {
                AppendCSVile(testDataPath, testCases);
            }

            ShowNetworkErrorWindow(learningResult);
            if(problem == ProblemType.Regression)
            {
                ShowRegressionWindow(network, trainingCases, testCases);
            }
            else if(problem == ProblemType.Classification)
            {
                ShowClassificationWindow(network, trainingCases, testCases);
            }
        }

        private void AppendCSVile(string path, CasesData data)
        {
            var output = new List<DenseVector>();

            for(int i = 0; i < data.CasesCount; i++)
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

        private void ShowRegressionWindow(NeuralNetwork neuralNetwork, CasesData trainingSet, CasesData testSet)
        {
            List<RegressionPoint> idealPoints = new List<RegressionPoint>();
            idealPoints.AddRange(MakeRegressionPointList(trainingSet));

            List<RegressionPoint> testPoints = new List<RegressionPoint>();            
            testPoints.AddRange(MakeRegressionPointList(testSet));

            Window regressionWindow = new RegressionWindow(testPoints, idealPoints); // trainingPoints are "ideal"
            regressionWindow.Show();
        }

        private IList<RegressionPoint> MakeRegressionPointList(CasesData dataSet)
        {
            IList<RegressionPoint> points = new List<RegressionPoint>();
            for (int i = 0; i < dataSet.CasesCount; i++)
            {
                points.Add(new RegressionPoint(dataSet.GetInput(i).At(0), dataSet.GetIdealOutput(i).At(0)));
            }

            return points;
        }

        private void ShowClassificationWindow(NeuralNetwork neuralNetwork, CasesData trainingSet, CasesData testSet)
        {
            List<ClassificationPoint> testPoints = new List<ClassificationPoint>();
            testPoints.AddRange(MakeClassificationPointList(testSet));

            List<ClassificationPoint> idealPoints = new List<ClassificationPoint>();
            idealPoints.AddRange(MakeClassificationPointList(trainingSet));

            Window classificationWindow = new ClassificationWindow(testPoints, idealPoints, trainingSet.ClassIndexes);
            classificationWindow.Show();
        }

        private IList<ClassificationPoint> MakeClassificationPointList(CasesData dataSet)
        {
            IList<ClassificationPoint> points = new List<ClassificationPoint>();
            for (int i = 0; i < dataSet.CasesCount; i++)
            {
                points.Add(new ClassificationPoint(dataSet.GetInput(i).At(0), 
                    dataSet.GetInput(i).At(1), (int)dataSet.GetClasificationOutput(i).At(0)));
            }

            return points;
        }
    }

    public enum YesNo
    {
        Yes,
        No,
    }

    public enum ActivationFunction
    {
        Unipolar,
        Bipolar
    }
}
