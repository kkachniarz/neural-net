using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningNN.DataSet
{
    [Obsolete("Has many leftovers from Part I. Use classess implementing IDataSet instead")]
    public class CasesData
    {
        public int ClassCount { get { return ClassIndexes.Count; } }
        public ProblemType ProblemType { get; private set; }
        public bool HasOutput { get; set; }
        public double MaxValue { get; set; }
        public double MinValue { get; set; }
        public int CasesCount { get { return cases.Count; } }
        public int BaseInputSize { get; private set; }
        public List<int> ClassIndexes { get; set; }

        private List<Tuple<DenseVector, DenseVector, DenseVector>> cases;
        private Normalizor normalizor;

        public CasesData(ProblemType problemType, List<DenseVector> input, List<DenseVector> output = null, 
            int outputLength = 1, int historyLength = 0)
        {
            if (input == null || input.Count == 0)
            {
                throw new ArgumentException("Input cannot be null or empty.");
            }

            HasOutput = output != null;

            if (HasOutput && input.Count != output.Count)
            {
                throw new ArgumentException("Input rows count should equal output rows count.");
            }

            ProblemType = problemType;

            if (ProblemType == ProblemType.Classification && HasOutput && output.First().Count != 1)
            {
                throw new ArgumentException("Classification problem should have only one defined output.");
            }

            if (outputLength <= 0)
            {
                throw new ArgumentException("regressionOutputLenght has to be positive.");
            }

            MinValue = double.MaxValue;
            MaxValue = double.MinValue;
            BaseInputSize = input[0].Count;

            ClassIndexes = new List<int>();
            cases = new List<Tuple<DenseVector, DenseVector, DenseVector>>();

            for (int i = 0; i < input.Count; i++)
            {
                CheckIfIsExtremum(input[i]);

                if (HasOutput)
                {
                    if (problemType == ProblemType.Regression)
                    {
                        CheckIfIsExtremum(output[i]);
                    }
                    else if (problemType == ProblemType.Classification)
                    {
                        UpdateClassCount((int)output[i][0]);
                    }
                }
            }

            for (int i = historyLength; i < input.Count; i++)
            {
                cases.Add(Tuple.Create(CreateInput(input, i, historyLength),
                    ProblemType == ProblemType.Classification ?
                        CreateClasyficationOutput(output, i, outputLength) :
                        CreateRegressionOutput(output, i, outputLength),
                        new DenseVector(outputLength)));
            }
        }

        private DenseVector CreateInput(List<DenseVector> inputs, int ind, int historyLength)
        {
            if (historyLength == 0)
            {
                return inputs[ind];
            }

            DenseVector historyAwareInput = new DenseVector(BaseInputSize * (historyLength + 1));
            for (int i = 0; i <= historyLength; i++)
            {
                inputs[ind - i].CopySubVectorTo(historyAwareInput, 0, i * BaseInputSize, BaseInputSize);
            }

            return historyAwareInput;
        }

        private DenseVector CreateRegressionOutput(List<DenseVector> output, int i, int outputLenght)
        {
            return (output == null) ? new DenseVector(outputLenght) : (DenseVector)output[i].Clone();
        }

        private DenseVector CreateClasyficationOutput(List<DenseVector> output, int i, int outputLenght)
        {
            if (output == null)
            {
                return new DenseVector(outputLenght);
            }

            var vector = new DenseVector(ClassCount);

            vector[ClassIndexes.IndexOf((int)output[i][0])] = 1.0;

            return vector;
        }

        private void UpdateClassCount(int value)
        {
            if (!ClassIndexes.Contains(value))
            {
                ClassIndexes.Add(value);
            }
        }

        private void CheckIfIsExtremum(DenseVector vector)
        {
            double actualMin = vector.Min();
            if (actualMin < MinValue)
            {
                MinValue = actualMin;
            }

            double actualMax = vector.Max();
            if (actualMax > MaxValue)
            {
                MaxValue = actualMax;
            }
        }

        public DenseVector GetInput(int i)
        {
            return cases[i].Item1;
        }

        public DenseVector GetIdealOutput(int i)
        {
            return cases[i].Item2;
        }

        public DenseVector GetNetworkAnswer(int i)
        {
            return cases[i].Item3;
        }

        public DenseVector GetClasificationOutput(int i)
        {
            var value = ClassIndexes[cases[i].Item2.MaximumIndex()]; ;
            return new DenseVector(new double[] { (double)value });
        }

        public void SaveNetworkAnswer(int i, Vector<double> answer)
        {
            if (answer.Count != cases[i].Item3.Count())
            {
                throw new ArgumentException();
            }

            answer.CopySubVectorTo(cases[i].Item3, 0, 0, answer.Count);
        }

        public void Permutate()
        {
            Random rnd = new Random(DateTime.Now.Millisecond);

            for (int i = 0; i < CasesCount; i++)
            {
                var index = rnd.Next(0, CasesCount);
                SwapCases(i, index);
            }
        }

        private void SwapCases(int i, int j)
        {
            var tmp = cases[i];
            cases[i] = cases[j];
            cases[j] = tmp;
        }

        public void Normalize(double _maxValueFrom, double _minValueFrom, double _maxValueTo, double _minValueTo)
        {
            normalizor = new Normalizor(_minValueFrom, _maxValueFrom, _minValueTo, _maxValueTo);

            for (int i = 0; i < cases.Count; i++)
            {
                GetInput(i).MapInplace(x => normalizor.Normalize(x));

                if (HasOutput && ProblemType == ProblemType.Regression)
                {
                    GetIdealOutput(i).MapInplace(x => normalizor.Normalize(x));
                }
            }
        }

        public void NormalizeBack(bool shouldNormalizeAnswer)
        {
            if (normalizor == null)
            {
                throw new ArgumentException();
            }

            IEnumerable<DenseVector> answers = cases.Select(x => x.Item3);
            for (int i = 0; i < cases.Count; i++)
            {
                GetInput(i).MapInplace(x => normalizor.NormalizeBack(x));

                if (ProblemType == ProblemType.Regression)
                {
                    GetIdealOutput(i).MapInplace(x => normalizor.NormalizeBack(x));
                    if (shouldNormalizeAnswer)
                    {
                        GetNetworkAnswer(i).MapInplace(x => normalizor.NormalizeBack(x));                        
                    }
                }
            }
        }

        public static void InitializeAllData(List<DenseVector> trainingData, List<DenseVector> testData, ProblemType problem,
            out CasesData trainingCases, out CasesData testCases)
        {
            int input = testData.First().Count();
            int output = trainingData.First().Count() - input;

            if (input < 1 || output < 1)
                throw new ArgumentException("Niepoprawne dane w plikach .csv");

            trainingCases = new CasesData(problem,
                trainingData.Select(x => x.CreateSubVector(0, input))
                    .ToList(),
                trainingData.Select(x => x.CreateSubVector(input, output))
                    .ToList(),
                output);

            testCases = new CasesData(problem, testData, null,
                (problem == ProblemType.Regression) ? output : trainingCases.ClassCount);

            testCases.ClassIndexes = trainingCases.ClassIndexes;
        }

        public static void InitializeForPrediction(List<DenseVector> dataSet, out CasesData trainingCases, out CasesData testCases, 
            int outputCount, int historyLength, float trainSetPercentage = 0.7f)
        {
            int inputCount = dataSet[0].Count() - outputCount;
            if (inputCount < 1 || outputCount < 1)
                throw new ArgumentException("Niepoprawne dane w plikach .csv");

            if(trainSetPercentage <= 0 || trainSetPercentage > 1)
            {
                throw new ArgumentException("Train Set Percentage should be between 0 and 1");
            }

            List<DenseVector> trainSet = new List<DenseVector>();
            List<DenseVector> testSet = new List<DenseVector>();
            int atIndex = 0;
            for(atIndex = 0; atIndex < dataSet.Count * trainSetPercentage; atIndex++)
            {
                trainSet.Add(dataSet[atIndex]);
            }

            if (trainSetPercentage == 1.0f) // special mode (testing on train set)
            {
                testSet = trainSet;
            }
            else
            {
                atIndex -= historyLength; // move back to have a "buffer" for historical data.
                while (atIndex < dataSet.Count)
                {
                    testSet.Add(dataSet[atIndex++]);
                }
            }

            trainingCases = new CasesData(ProblemType.Regression,
                trainSet.Select(x => x.CreateSubVector(0, inputCount))
                    .ToList(),
                trainSet.Select(x => x.CreateSubVector(inputCount, outputCount))
                    .ToList(),
                outputCount,
                historyLength);

            testCases = new CasesData(ProblemType.Regression,
                testSet.Select(x => x.CreateSubVector(0, inputCount)).ToList(),
                testSet.Select(x => x.CreateSubVector(inputCount, outputCount))
                    .ToList(), 
                outputCount,
                historyLength);
        }
    }
}
