# About Tiny Dataset

The tiny dataset is meant to help you debug your code using an almost minimal example. The datasets `tinyTrain.csv` and `tinyValidation.csv` are in a format similar to the small, medium, and large datasets, except that they only have 2 samples with 5 features. The corresponding detailed output `tinyOutput.txt` contains the results one would expect from a correctly implemented neural network, with the following command line arguments:

```
python3 neuralnet.py tinyTrain.csv tinyValidation.csv _ _ _ 1 4 2 0.1
```

Note that this is *only one possible* output from a correct implementation. It is perfectly fine if yours are different from this output in some aspect, e.g., different places to put the bias terms, different derivative layouts, etc., as long as you are consistent and can verify your correctness.
