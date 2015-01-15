using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpNN;
using RecursiveNN;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

namespace NetworkToolbox
{
    /// <summary>
    /// Possible functionality: loading a network, training it further, testing on specified data set etc. 
    /// We can train a good network and then load it and present how it works.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            IFormatter formatter = new BinaryFormatter();
            Stream stream = new FileStream("neural.bin", FileMode.Open, FileAccess.Read, FileShare.Read);
            INetwork obj = (INetwork)formatter.Deserialize(stream);
            Console.WriteLine(obj is NeuralNetwork);
            stream.Close();
        }
    }
}
