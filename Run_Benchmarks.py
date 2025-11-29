import numpy as np
from sklearn.metrics import get_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import tqdm
import pandas as pd
import time

from gen_data import datasources

print("Checking dataset loaders...")
for name,datasource in datasources.items():
    t0 = time.time()
    print("%-50s"%(name+"... "), end="")
    datasource = datasource()
    datasource.take(100)
    datasource.generate_drift()
    datasource.take(100)
    t1 = time.time()
    print("OK (%i)"%(t1-t0))

def generate_stream(name, config):
    datasource = datasources[name]()
    batch_size = 200
    stream,dd_true = [], []
    for segm_type in f"---{config}-":
        v,r = datasource.generate_drift(segm_type)
        dd_true.append( (v+2*r)*np.ones(batch_size) )
        segment = datasource.take(batch_size)
        stream.append(segment)
        
    data = np.concatenate(list(map(lambda x:x[0],stream)), axis=0)
    label = np.concatenate(list(map(lambda x:x[1],stream)), axis=0)
    
    if not (
        min([np.unique(label[s]).size for s in [range(0,400),range(400,600),range(600,800)]]) > 1 and 
        min([np.unique(label[s],return_counts=True)[1].min() for s in [range(0,400),range(400,600),range(600,800)]]) > 10):
     return None
        
    return {"train_X_s": data[:400],
            "train_y_s": label[:400],
            "test_X_n": data[400:600],
            "test_y_n": label[400:600],
            "test_X_d": data[600:800],
            "test_y_d": label[600:800],
            "train_X_e": np.vstack((data[:400],data[800:])),
            "train_y_e": np.hstack((label[:400],label[800:])),
            "drifts": np.hstack(dd_true),
            "source name": name,
            "loader": datasource.get_info(),
            "drift_type": config}

models = [("kNN",KNeighborsClassifier()),
          ("ET",ExtraTreesClassifier()),
          ("RF",RandomForestClassifier()),
          ("DT",DecisionTreeClassifier()),
          ("SVM",LinearSVC(max_iter=5000)),
          ("Perc",Perceptron(max_iter=5000)),
          ("LR",LogisticRegressionCV(max_iter=5000)),
          ("MLP",MLPClassifier(max_iter=5000))
         ]
score_functions = ["accuracy","f1","matthews_corrcoef","roc_auc_ovr"]

def scorer(name, mode, X, y):
    try:
        return get_scorer(name)(model, X, y)
    except:
        return np.nan

n_reps = 500

results = []
t0 = time.time()
for dataset_id, (config,name) in enumerate(tqdm.tqdm(n_reps*[(config,name) for config in ["v","r"] for name in datasources.keys()])):
    dataset = generate_stream(name, config)
    if dataset is not None:
        for trainT in ["s","e"]:
            for model_name, model in models:
                model = clone(model)
                try:
                    model.fit(dataset[f"train_X_{trainT}"],dataset[f"train_y_{trainT}"])
                except:
                    model = None
                results.append({"loader": dataset["loader"], 
                                "source name": dataset["source name"], 
                                "exp_id": dataset_id,
                                "drift type": dataset["drift_type"],
                                "train type": trainT,
                                "model": model_name,
                                "drift state": False,
                                "score": {name: scorer(name, model, dataset["test_X_n"], dataset["test_y_n"]) for name in score_functions}
                            })
                results.append({"loader": dataset["loader"], 
                                "source name": dataset["source name"], 
                                "exp_id": dataset_id,
                                "drift type": dataset["drift_type"],
                                "train type": trainT,
                                "model": model_name,
                                "drift state": True,
                                "score": {name: scorer(name, model, dataset["test_X_d"], dataset["test_y_d"]) for name in score_functions}
                            })
    if time.time()-t0 > 5*60:
        try:
            pd.DataFrame(results).to_pickle(f"results.pkl.xz")
            t0 = time.time()
        except Exception as e:
            print("failed to save due to "+str(e))
while True:
    try:
        pd.DataFrame(results).to_pickle(f"results.pkl.xz")
        break
    except Exception as e:
        print("failed  to save due to"+str(e))
        time.sleep(1)
