using LearningNN;
using LearningNN.DataSet;
using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.Win32;
using Neural_Network;
using Neural_Network.Plotting;
using OxyPlot;
using OxyPlot.Wpf;
using RecursiveNN;
using SharpNN;
using SharpNN.ActivationFunctions;
using SharpNN.Statistics;
using Shell.Containers;
using Shell.Enums;
using Shell.Plotting;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
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

namespace Shell
{
    public partial class MainWindow : Window, ILearningStatus
    {
        public bool IsReady { get { return csvLines != null; } }

        private const double ERROR_SCALE = 1000.0;
        private const double DISCARD_FACTOR = 0.0;
        private const int DISPLAY_LIMIT = 10;
        private int runCounter = 0;
        private int runsPerSettings = 1;         
        private bool plotAgainstInput = false;
        private string resultsDirectoryPath;

        private Dictionary<LearningSettings, List<SingleRunReport>> resultsBySettings = new Dictionary<LearningSettings, List<SingleRunReport>>();
        
        ReportingOptions reportingOptions;
        List<DenseVector> csvLines;
        List<LearningSettings> settingsToRun = new List<LearningSettings>();
        IDataSet testDataSet;
        IDataSet trainDataSet;
        ILearningStrategy learningStrategy;
        private readonly BackgroundWorker worker = new BackgroundWorker();

        string dataSetPath;
        string parametersFileName = "";

        public MainWindow()
        {
            InitializeComponent();

            NetworkTypeCombobox.ItemsSource = Enum.GetValues(typeof(NetworkType)).Cast<NetworkType>();
            ActivationCombobox.ItemsSource = Enum.GetValues(typeof(ActivationType)).Cast<ActivationType>();
            BiasCombobox.ItemsSource = Enum.GetValues(typeof(YesNo)).Cast<YesNo>();

            ActivationCombobox.SelectedIndex = 0;
            BiasCombobox.SelectedIndex = 0;
        }

        private void ReadParameters(object sender, RoutedEventArgs e)
        {
            string shortName;
            OpenFileDialog csvDlg = new OpenFileDialog();
            csvDlg.DefaultExt = ".txt";
            csvDlg.Filter = "TXT documents (.txt)|*.txt";
            csvDlg.Title = "Select a .txt file containing network parameters";
            string paramsPath = ReadFile(out shortName, csvDlg);
            parametersFileName = shortName;

            if (paramsPath == null)
                return;

            settingsToRun = FileManager.RetrieveParameters(paramsPath);
            LoadParametersLabel.Content = shortName;
            // allow to unload parameters from UI
        }

        private void ReadDataSet(object sender, RoutedEventArgs e)
        {
            string shortName;
            OpenFileDialog csvDlg = new OpenFileDialog();
            csvDlg.DefaultExt = ".csv";
            csvDlg.Filter = "CSV documents (.csv)|*.csv";
            csvDlg.Title = "Select proper CSV file";
            dataSetPath = ReadFile(out shortName, csvDlg);

            if (dataSetPath == null)
                return;

            csvLines = FileManager.ReadDataFromCSV(dataSetPath);
            TrainSetLabel.Content = shortName;

            if (IsReady)
            {
                StartButton.IsEnabled = true;
            }
        }

        private string ReadFile(out string shortName, OpenFileDialog dialog)
        {
            var result = dialog.ShowDialog();

            if (result == true)
            {
                var fileName = dialog.FileName.ToString();
                shortName = dialog.SafeFileName;
                return fileName;
            }

            shortName = null;
            return null;
        }

        private void StartButtonClick(object sender, RoutedEventArgs e)
        {
            StartButton.IsEnabled = false;

            worker.DoWork += worker_DoWork;
            worker.RunWorkerCompleted += worker_RunWorkerCompleted;
            worker.ProgressChanged += worker_ProgressChanged;

            runCounter = 0;
            runsPerSettings = int.Parse(RunsTextBox.Text);
            string[] layersText = LayersTextBox.Text.Split(new string[] { ",", " ", "-", "_", "." }, StringSplitOptions.RemoveEmptyEntries);
            bool bias = (YesNo)BiasCombobox.SelectedItem == YesNo.Yes;

            int pcaDimensions = 0;
            int.TryParse(PCA.Text, out pcaDimensions); // TODO: test

            float trainSetPercentage = float.Parse(TrainSetPercentage.Text, CultureInfo.InvariantCulture);
            int outputCount = int.Parse(OutputCount.Text);
            NetworkType networkType = (NetworkType)NetworkTypeCombobox.SelectedItem;
            int ctsPrevValuesCount = 1; // int.Parse(CTSPreviousValues.Text); <- obsolete, never used (CT series depend only on 1 previous value...)
            PartIIProblemType problemType = csvLines[0].Count == 1 ? PartIIProblemType.CTS : PartIIProblemType.Stock;
            reportingOptions = GetReportingOptions();

            if (settingsToRun.Count == 0)
            {
                settingsToRun.Add(GetLearningSettingsFromUI());
            }

            if (problemType == PartIIProblemType.CTS)
            {
                plotAgainstInput = true;
            }

            ConfirmReportingSettings();

            EngineInitData eid = new EngineInitData();

            eid.ErrorScale = ERROR_SCALE;
            eid.DiscardWorstFactor = DISCARD_FACTOR;
            eid.TrainSetPercentage = trainSetPercentage;

            eid.CtsPrevValuesCount = 1; // always 1
            eid.OutputCount = outputCount;
            eid.InputCount = problemType == PartIIProblemType.CTS? 1 : csvLines[0].Count - outputCount;
            if (pcaDimensions > 0)
            {
                eid.InputCount = Math.Min(pcaDimensions, eid.InputCount);
            }

            eid.PcaDimensions = pcaDimensions;
            eid.RunsPerSettings = runsPerSettings;

            eid.UseBiases = bias;
            eid.PlotAgainstInput = plotAgainstInput;

            eid.NetworkType = networkType;
            eid.ProblemType = problemType;

            eid.ReportingOptions = reportingOptions;

            eid.HiddenNeuronCounts = layersText.Select(s => int.Parse(s)).ToList(); // TODO: test
            eid.CsvLines = csvLines;
            eid.SettingsToRun = settingsToRun;

            eid.ResultsDirectoryPath = resultsDirectoryPath;
            eid.DataSetName = dataSetPath;
            eid.ParametersFileName = parametersFileName;

            Engine engine = new Engine(eid);
            // TODO: in a worker
            engine.Run();
         
            StartButton.IsEnabled = true;
        }

        void worker_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            throw new NotImplementedException();
        }

        void worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            throw new NotImplementedException();
        }

        void worker_DoWork(object sender, DoWorkEventArgs e)
        {
            throw new NotImplementedException();
        }

        private void ConfirmReportingSettings()
        {
            if (reportingOptions.ShouldDisplay && settingsToRun.Count * runsPerSettings >= DISPLAY_LIMIT)
            {
                MessageBoxResult result = MessageBox.Show("Display results for " + (settingsToRun.Count * runsPerSettings).ToString() + "?"
                    , "Confirmation", MessageBoxButton.YesNo, MessageBoxImage.Question);
                if (result == MessageBoxResult.No)
                {
                    reportingOptions.ShouldDisplay = false;
                }
            }
        }

        private ReportingOptions GetReportingOptions()
        {
            ReportingOptions options = new ReportingOptions();
            options.ShouldDisplay = ShowPlotsCheckbox.IsChecked.Value;
            options.ShouldSave = SavePlotsCheckbox.IsChecked.Value;
            return options;
        }

        private LearningSettings GetLearningSettingsFromUI()
        {
            LearningSettings lSettings = new LearningSettings();
            lSettings.MaxIterations = int.Parse(MaxIterations.Text);
            lSettings.BadIterations = BadIterations.Text == "" ? lSettings.MaxIterations : int.Parse(BadIterations.Text);
            lSettings.LearningRate = double.Parse(LearningRate.Text);
            lSettings.Momentum = double.Parse(Momentum.Text);
            lSettings.ValidationSetSize = 0.2f;
            lSettings.Activation = ((ActivationType)ActivationCombobox.SelectedItem == ActivationType.Bipolar) ?
                (IActivation)new BipolarTanhActivation() : new UnipolarSigmoidActivation();

            return lSettings;
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


        public void SetStatusText(string text)
        {           
            this.Title = string.Format("{0} / {1}: {2}", runCounter, settingsToRun.Count * runsPerSettings, text);
        }

    }
}
