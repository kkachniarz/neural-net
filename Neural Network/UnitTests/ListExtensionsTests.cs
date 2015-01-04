using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using Shell;

namespace UnitTests
{
    [TestClass]
    public class ListExtensionsTests
    {
        [TestMethod]
        public void RemoveHighestValuesRemovesCorrectNumberOfElements()
        {            
            List<double> myList = new List<double>() { 1, 6, 4, 3, 2, 87, 54, 8, -34 };
            int initialCount = myList.Count;
            int countToRemove = 3;

            myList.RemoveHighestValues(x => x, countToRemove);
            Assert.AreEqual(initialCount - countToRemove, myList.Count);            
        }

        [TestMethod]
        public void RemoveHighestValuesRemovesTopElements()
        {
            List<double> myList = new List<double>() { 1, 6, 4, 3, 2, 87, 54, 8, -34 };
            int initialCount = myList.Count;
            int countToRemove = 3;

            myList.RemoveHighestValues(x => x, countToRemove);
            Assert.IsFalse(myList.Contains(87));
            Assert.IsFalse(myList.Contains(54));
            Assert.IsFalse(myList.Contains(8));
        }

        [TestMethod]
        public void RemoveHighestValuesDoesntSort()
        {
            List<double> myList = new List<double>() { 1, 6, 4, 3, 2, 87, 54, 8, -34 };
            int initialCount = myList.Count;
            int countToRemove = 3;

            myList.RemoveHighestValues(x => x, countToRemove);
            Assert.AreEqual(1, myList[0]);
            Assert.AreEqual(6, myList[1]);
            Assert.AreEqual(4, myList[2]);
        }
    }
}
