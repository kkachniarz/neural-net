using System;
using System.Linq;

namespace LearningNN
{
    public static class PCA
    {
        public static void Run(IDataSet set, int destinationInputLenght)
        {
            if (set == null 
                || set.EnumeratePatterns().First() == null 
                || set.EnumeratePatterns().First().Input == null)
                throw new ArgumentException();

            int inputLenght = set.EnumeratePatterns().First().Input.Count;

            if (destinationInputLenght >= inputLenght)
                return;

            // TODO:
            // Impelment PCA logic
        }
    }
}
