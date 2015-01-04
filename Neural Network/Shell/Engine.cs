using LearningNN.DataSet;
using LearningNN.Learning;
using MathNet.Numerics.LinearAlgebra.Double;
using OxyPlot;
using Shell.Containers;
using Shell.Plotting;
using LearningNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpNN;
using SharpNN.Statistics;
using System.IO;
using Shell.Enums;
using RecursiveNN;
using OxyPlot.Wpf;
using System.Windows;
using Neural_Network.Plotting;
using System.ComponentModel;
using System.Windows.Threading;

namespace Shell
{
    /// <summary>
    /// Executes the learning N times, where N = runs_per_settings * how_many_different_settings_we_have,
    /// and groups the results by learning settings.
    /// </summary>
    public class Engine : IStatusReporter
    {
        private EngineInitData eid;
        public Dictionary<LearningSettings, List<SingleRunReport>> resultsBySettings 
            = new Dictionary<LearningSettings, List<SingleRunReport>>();
        private int runCounter;
        private int discardCount;

        private IDataSet trainSet;
        private IDataSet testSet;
        private ILearningStrategy learningStrategy;
        private MainWindow mainWindow;

        public Engine(EngineInitData engineInitData, MainWindow mainWindow)
        {
            this.eid = engineInitData;
            this.mainWindow = mainWindow;
            discardCount = (int)(eid.DiscardWorstFactor * (double)eid.RunsPerSettings);
        }

        public EngineResult Run()
        {
            foreach (LearningSettings learningSettings in eid.SettingsToRun)
            {
                resultsBySettings[learningSettings] = new List<SingleRunReport>();
                for (int i = 0; i < eid.RunsPerSettings; i++)
                {
                    runCounter++;
                    List<int> layersVal = BuildLayersVal();
                    BuildDataSet(layersVal);
                    INetwork network = CreateNetwork(learningSettings, layersVal);

                    NormalizeData(network, trainSet, testSet);
                    CheckIfPerformPCA(network);
                    learningStrategy = new VSetLearningStrategy(learningSettings);

                    var learningResult = BackpropagationManager.Run(network, trainSet, testSet,
                        learningStrategy, this);

                    NormalizeDataBack(network, trainSet, testSet);
                    resultsBySettings[learningSettings].Add(new SingleRunReport(
                        network, layersVal, DateTime.Now, learningResult, trainSet, testSet));
                }

                resultsBySettings[learningSettings].RemoveHighestValues(
                    r => r.LearningResult.TestSetError, discardCount);
            }

            EngineResult result = new EngineResult();
            result.ResultsBySettings = resultsBySettings;
            result.Eid = eid;
            result.WorstDiscardedCount = discardCount;
            return result;
        }

        private List<int> BuildLayersVal()
        {
            List<int> ret = new List<int>();
            ret.Add(eid.InputCount);
            foreach (int neuronCount in eid.HiddenNeuronCounts)
            {
                ret.Add(neuronCount); //TODO: Later layer counts should be configurable in params file / learning settings
            }

            ret.Add(eid.OutputCount);
            return ret;
        }

        private void BuildDataSet(List<int> layersVal)
        {
            if (eid.ProblemType == PartIIProblemType.CTS)
            {
                InitCTS(layersVal, eid.TrainSetPercentage);
            }
            else
            {
                InitStock(layersVal, eid.TrainSetPercentage);
            }
        }

        private INetwork CreateNetwork(LearningSettings learningSettings, List<int> layersVal)
        {
            INetwork network = null;
            switch (eid.NetworkType)
            {
                case NetworkType.MLP:
                    network = new NeuralNetwork(learningSettings.Activation, eid.UseBiases, layersVal.ToArray());
                    break;
                case NetworkType.Jordan:
                    network = new RecursiveNetwork(RecursiveNetwork.Type.Jordan,
                    learningSettings.Activation, eid.UseBiases, layersVal[0], layersVal[1], layersVal[2]);
                    break;
                case NetworkType.Elman:
                    network = new RecursiveNetwork(RecursiveNetwork.Type.Elman,
                    learningSettings.Activation, eid.UseBiases, layersVal[0], layersVal[1], layersVal[2]);
                    break;
            }

            return network;
        }

        private void InitCTS(List<int> layersVal, float trainSetPercentage)
        {
            int historyLength = 1; // always historyLength = 1
            int trainSetEndIndex = (int)(trainSetPercentage * eid.CsvLines.Count);
            List<DenseVector> chaoticValues = eid.CsvLines; // no need for further parsing

            List<DenseVector> trainValues = chaoticValues.ExtractList(0, trainSetEndIndex);
            List<DenseVector> testValues = chaoticValues.ExtractList(trainSetEndIndex, chaoticValues.Count);

            trainSet = new ChaoticDataSet(trainValues, historyLength, 0);
            if (trainSetEndIndex >= chaoticValues.Count - 1)
            {
                testSet = trainSet.Clone();
            }
            else
            {
                testSet = new ChaoticDataSet(testValues, historyLength, trainSetEndIndex);
            }
        }

        private void InitStock(List<int> layersVal, float trainSetPercentage)
        {
            int trainSetEndIndex = (int)(trainSetPercentage * eid.CsvLines.Count);
            List<DenseVector> allInputs = eid.CsvLines.Select(v => v.CreateSubVector(0, eid.InputCount)).ToList();
            List<DenseVector> allOutputs = eid.CsvLines.Select(v => v.CreateSubVector(eid.InputCount, eid.OutputCount)).ToList();

            trainSet = new StockDataSet(allInputs.ExtractList(0, trainSetEndIndex),
                allOutputs.ExtractList(0, trainSetEndIndex), 0);

            if (trainSetEndIndex >= allInputs.Count - 1)
            {
                testSet = trainSet.Clone();
            }
            else
            {
                testSet = new StockDataSet(allInputs.ExtractList(trainSetEndIndex, eid.CsvLines.Count),
                    allOutputs.ExtractList(trainSetEndIndex, eid.CsvLines.Count), trainSetEndIndex);
            }
        }

        private void CheckIfPerformPCA(INetwork network)
        {
            if (eid.PcaDimensions > 0)
            {
                LearningNN.PCA.Run(trainSet, eid.PcaDimensions, network.Activation.MinValue, network.Activation.MaxValue);
                LearningNN.PCA.Run(testSet, eid.PcaDimensions, network.Activation.MinValue, network.Activation.MaxValue);
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

        public void UpdateStatus(string text)
        {
            string message = string.Format("{0}/{1}: {2}", runCounter, 
                eid.RunsPerSettings * eid.SettingsToRun.Count, text);
            mainWindow.Dispatcher.BeginInvoke((Action<string>)mainWindow.UpdateStatus,
                DispatcherPriority.Background, message);
        }
    }
}
