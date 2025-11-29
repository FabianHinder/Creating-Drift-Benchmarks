# Real vs. Virtual Drift: Creating Realistic Stream Learning Benchmarks

Experimental code of conference paper. (under review)

## Abstract
Concept drift -- changes in the data distribution over time -- is a central challenge in stream learning. However, existing benchmarks either lack controlled drift or fail to capture the characteristics of real-world data. We propose a pipeline for constructing verifiable and realistic drift, enabling more systematic evaluation of stream learning algorithms. Here, we pay special attention to controlling both real and virtual drift. To underscore the relevance of our contribution, we analyze the effects of real and virtual drift on both real-world and synthetic data streams using our method, revealing a substantial mismatch between the two setups.

## Requirements
* Python 
* Numpy, SciPy, Pandas, Matplotlib
* scikit-learn, river

## Usage
Run all experiments via the `Run_Benchmarks.py` script. One analysis of the resutls is presented in `Analyze.ipynb`.

## Cite

Cite out paper

```
currently under review
```

## Real-World Streams
* *Nebraska Weather* [Original source](https://users.rowan.edu/~polikar/nse.html)
* *Electricity market dataset* [Original source](http://www.inescporto.pt/~jgama/ales/ales_5.html)
* *Forest Cover Type* [Original source](https://archive.ics.uci.edu/ml/datasets/Covertype)
* *HTTP* dataset of the KDD 1999 cup. [Original source](https://odds.cs.stonybrook.edu/http-kddcup99-dataset/)
* *Credit Card* [Source](https://maxhalford.github.io/files/datasets/creditcardfraud.zip)

## License
This code has a MIT license. 

Datasets are not included in the license and only provided for conviniance and may fall under a different licens. 
