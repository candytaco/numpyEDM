"""
The prediction core: array-in, array-out functions and the drivers built on them.

- Predictors: SimplexPredict, SimplexGenerate, SMapPredict, SMapGenerate
- Multiview: MultiviewPredict
- ConvergentCrossMap: cross-map skill of many source columns across training-subset sizes
- MDE: greedy variable selection
- Setup: turning X/Y arrays into training pairs and test states
- _core: the tensor kernels every predictor shares
- Results: the result records and ResultsIO
"""
