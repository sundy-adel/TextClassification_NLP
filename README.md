```python
%pip install mlxtend --upgrade
%pip install tensorflow
%pip install --upgrade tensorflow
%pip install xgboost
```

    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    

    Requirement already satisfied: mlxtend in c:\users\lenovo\anaconda3\lib\site-packages (0.22.0)
    Requirement already satisfied: scipy>=1.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from mlxtend) (1.7.3)
    Requirement already satisfied: numpy>=1.16.2 in c:\users\lenovo\anaconda3\lib\site-packages (from mlxtend) (1.22.4)
    Requirement already satisfied: pandas>=0.24.2 in c:\users\lenovo\anaconda3\lib\site-packages (from mlxtend) (1.4.2)
    Requirement already satisfied: scikit-learn>=1.0.2 in c:\users\lenovo\anaconda3\lib\site-packages (from mlxtend) (1.0.2)
    Requirement already satisfied: matplotlib>=3.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from mlxtend) (3.5.1)
    Requirement already satisfied: joblib>=0.13.2 in c:\users\lenovo\anaconda3\lib\site-packages (from mlxtend) (1.1.0)
    Requirement already satisfied: setuptools in c:\users\lenovo\anaconda3\lib\site-packages (from mlxtend) (61.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\lenovo\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\lenovo\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\lenovo\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (1.3.2)
    Requirement already satisfied: packaging>=20.0 in c:\users\lenovo\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (21.3)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\lenovo\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (9.5.0)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (3.0.4)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\lenovo\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\lenovo\anaconda3\lib\site-packages (from pandas>=0.24.2->mlxtend) (2021.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from scikit-learn>=1.0.2->mlxtend) (2.2.0)
    Requirement already satisfied: six>=1.5 in c:\users\lenovo\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib>=3.0.0->mlxtend) (1.15.0)
    Requirement already satisfied: tensorflow in c:\users\lenovo\anaconda3\lib\site-packages (2.9.1)
    Note: you may need to restart the kernel to use updated packages.Requirement already satisfied: absl-py>=1.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.2.0)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: flatbuffers<2,>=1.12 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.12)
    
    

    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    

    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (0.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (3.6.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.1.2)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (14.0.6)
    Requirement already satisfied: numpy>=1.20 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.22.4)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (3.3.0)
    Requirement already satisfied: packaging in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (21.3)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (3.19.6)
    Requirement already satisfied: setuptools in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (61.2.0)
    Requirement already satisfied: six>=1.12.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.15.0)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.1.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (3.7.4.3)
    Requirement already satisfied: wrapt>=1.11.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.12.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (0.26.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.54.2)
    Requirement already satisfied: tensorboard<2.10,>=2.9 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (2.9.1)
    Requirement already satisfied: tensorflow-estimator<2.10.0,>=2.9.0rc0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (2.9.0)
    Requirement already satisfied: keras<2.10.0,>=2.9.0rc0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (2.9.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\lenovo\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
    Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.19.1)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (0.4.6)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (3.3.4)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.27.1)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (0.6.1)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (1.8.1)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.0.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\lenovo\anaconda3\lib\site-packages (from packaging->tensorflow) (3.0.4)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (4.2.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (4.7.2)
    Requirement already satisfied: urllib3<2.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (1.26.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow) (1.3.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (3.3)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\lenovo\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (0.4.8)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow) (3.2.0)
    Requirement already satisfied: tensorflow in c:\users\lenovo\anaconda3\lib\site-packages (2.9.1)
    Collecting tensorflow
      Using cached tensorflow-2.12.0-cp39-cp39-win_amd64.whl (1.9 kB)
    Collecting tensorflow-intel==2.12.0 (from tensorflow)
      Using cached tensorflow_intel-2.12.0-cp39-cp39-win_amd64.whl (272.8 MB)
    Requirement already satisfied: absl-py>=1.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.2.0)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.6.3)
    Collecting flatbuffers>=2.0 (from tensorflow-intel==2.12.0->tensorflow)
      Using cached flatbuffers-23.5.26-py2.py3-none-any.whl (26 kB)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (3.6.0)
    Requirement already satisfied: jax>=0.3.15 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.4.11)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (14.0.6)
    Requirement already satisfied: numpy<1.24,>=1.22 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.22.4)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (3.3.0)
    Requirement already satisfied: packaging in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (21.3)
    Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow-intel==2.12.0->tensorflow)
      Using cached protobuf-4.23.2-cp39-cp39-win_amd64.whl (422 kB)
    Requirement already satisfied: setuptools in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (61.2.0)
    Requirement already satisfied: six>=1.12.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.15.0)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.1.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (3.7.4.3)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.12.1)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (1.54.2)
    Collecting tensorboard<2.13,>=2.12 (from tensorflow-intel==2.12.0->tensorflow)
      Using cached tensorboard-2.12.3-py3-none-any.whl (5.6 MB)
    Collecting tensorflow-estimator<2.13,>=2.12.0 (from tensorflow-intel==2.12.0->tensorflow)
      Using cached tensorflow_estimator-2.12.0-py2.py3-none-any.whl (440 kB)
    Collecting keras<2.13,>=2.12.0 (from tensorflow-intel==2.12.0->tensorflow)
      Using cached keras-2.12.0-py2.py3-none-any.whl (1.7 MB)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow-intel==2.12.0->tensorflow) (0.26.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\lenovo\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.12.0->tensorflow) (0.37.1)
    Requirement already satisfied: ml-dtypes>=0.1.0 in c:\users\lenovo\anaconda3\lib\site-packages (from jax>=0.3.15->tensorflow-intel==2.12.0->tensorflow) (0.1.0)
    Requirement already satisfied: scipy>=1.7 in c:\users\lenovo\anaconda3\lib\site-packages (from jax>=0.3.15->tensorflow-intel==2.12.0->tensorflow) (1.7.3)
    Requirement already satisfied: importlib-metadata>=4.6 in c:\users\lenovo\anaconda3\lib\site-packages (from jax>=0.3.15->tensorflow-intel==2.12.0->tensorflow) (4.11.3)
    Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.19.1)
    Collecting google-auth-oauthlib<1.1,>=0.5 (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow)
      Using cached google_auth_oauthlib-1.0.0-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (3.3.4)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.27.1)
    Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow)
      Using cached tensorboard_data_server-0.7.0-py3-none-any.whl (2.4 kB)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.0.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\lenovo\anaconda3\lib\site-packages (from packaging->tensorflow-intel==2.12.0->tensorflow) (3.0.4)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (4.2.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (4.7.2)
    Requirement already satisfied: urllib3<2.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (1.26.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (1.3.1)
    Requirement already satisfied: zipp>=0.5 in c:\users\lenovo\anaconda3\lib\site-packages (from importlib-metadata>=4.6->jax>=0.3.15->tensorflow-intel==2.12.0->tensorflow) (3.7.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (3.3)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\lenovo\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (0.4.8)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow-intel==2.12.0->tensorflow) (3.2.0)
    Installing collected packages: flatbuffers, tensorflow-estimator, tensorboard-data-server, protobuf, keras, google-auth-oauthlib, tensorboard, tensorflow-intel, tensorflow
      Attempting uninstall: flatbuffers
        Found existing installation: flatbuffers 1.12
        Uninstalling flatbuffers-1.12:
          Successfully uninstalled flatbuffers-1.12
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.9.0
        Uninstalling tensorflow-estimator-2.9.0:
          Successfully uninstalled tensorflow-estimator-2.9.0
      Attempting uninstall: tensorboard-data-server
        Found existing installation: tensorboard-data-server 0.6.1
        Uninstalling tensorboard-data-server-0.6.1:
          Successfully uninstalled tensorboard-data-server-0.6.1
      Attempting uninstall: protobuf
        Found existing installation: protobuf 3.19.6
        Uninstalling protobuf-3.19.6:
          Successfully uninstalled protobuf-3.19.6
      Attempting uninstall: keras
        Found existing installation: keras 2.9.0
        Uninstalling keras-2.9.0:
          Successfully uninstalled keras-2.9.0
      Attempting uninstall: google-auth-oauthlib
        Found existing installation: google-auth-oauthlib 0.4.6
        Uninstalling google-auth-oauthlib-0.4.6:
          Successfully uninstalled google-auth-oauthlib-0.4.6
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.9.1Note: you may need to restart the kernel to use updated packages.
    
    

    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: 'C:\\Users\\LENOVO\\anaconda3\\Lib\\site-packages\\tensorflow\\python\\_pywrap_tensorflow_internal.pyd'
    Consider using the `--user` option or check the permissions.
    
    

        Uninstalling tensorboard-2.9.1:
          Successfully uninstalled tensorboard-2.9.1
    Requirement already satisfied: xgboost in c:\users\lenovo\anaconda3\lib\site-packages (1.7.5)
    Requirement already satisfied: numpy in c:\users\lenovo\anaconda3\lib\site-packages (from xgboost) (1.22.4)
    Requirement already satisfied: scipy in c:\users\lenovo\anaconda3\lib\site-packages (from xgboost) (1.7.3)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    

# Importing the Libraries


```python
import math
import nltk
import pandas as pd 
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import gutenberg
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from mlxtend.evaluate import bias_variance_decomp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
```


```python
%pip install tensorflow
import tensorflow as tf
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences
```

    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    google-api-core 1.25.1 requires google-auth<2.0dev,>=1.21.1, but you have google-auth 2.19.1 which is incompatible.
    google-cloud-core 1.7.1 requires google-auth<2.0dev,>=1.24.0, but you have google-auth 2.19.1 which is incompatible.
    google-cloud-storage 1.31.0 requires google-auth<2.0dev,>=1.11.0, but you have google-auth 2.19.1 which is incompatible.
    streamlit 1.12.0 requires typing-extensions>=3.10.0.0, but you have typing-extensions 3.7.4.3 which is incompatible.
    

    Requirement already satisfied: tensorflow in c:\users\lenovo\anaconda3\lib\site-packages (2.9.1)
    Requirement already satisfied: absl-py>=1.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.2.0)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.6.3)
    Collecting flatbuffers<2,>=1.12 (from tensorflow)
      Using cached flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (0.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (3.6.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.1.2)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (14.0.6)
    Requirement already satisfied: numpy>=1.20 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.22.4)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (3.3.0)
    Requirement already satisfied: packaging in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (21.3)
    Collecting protobuf<3.20,>=3.9.2 (from tensorflow)
    Note: you may need to restart the kernel to use updated packages.  Using cached protobuf-3.19.6-cp39-cp39-win_amd64.whl (895 kB)
    Requirement already satisfied: setuptools in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (61.2.0)
    Requirement already satisfied: six>=1.12.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.15.0)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.1.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (3.7.4.3)
    Requirement already satisfied: wrapt>=1.11.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.12.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (0.26.0)
    
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorflow) (1.54.2)
    Collecting tensorboard<2.10,>=2.9 (from tensorflow)
      Using cached tensorboard-2.9.1-py3-none-any.whl (5.8 MB)
    Collecting tensorflow-estimator<2.10.0,>=2.9.0rc0 (from tensorflow)
      Using cached tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
    Collecting keras<2.10.0,>=2.9.0rc0 (from tensorflow)
      Using cached keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\lenovo\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
    Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.19.1)
    Collecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.10,>=2.9->tensorflow)
      Using cached google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (3.3.4)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.27.1)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard<2.10,>=2.9->tensorflow)
      Using cached tensorboard_data_server-0.6.1-py3-none-any.whl (2.4 kB)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (1.8.1)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\lenovo\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.0.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\lenovo\anaconda3\lib\site-packages (from packaging->tensorflow) (3.0.4)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (4.2.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (4.7.2)
    Requirement already satisfied: urllib3<2.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (1.26.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\lenovo\anaconda3\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow) (1.3.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\lenovo\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (3.3)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\lenovo\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow) (0.4.8)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow) (3.2.0)
    Installing collected packages: keras, flatbuffers, tensorflow-estimator, tensorboard-data-server, protobuf, google-auth-oauthlib, tensorboard
      Attempting uninstall: keras
        Found existing installation: keras 2.12.0
        Uninstalling keras-2.12.0:
          Successfully uninstalled keras-2.12.0
      Attempting uninstall: flatbuffers
        Found existing installation: flatbuffers 23.5.26
        Uninstalling flatbuffers-23.5.26:
          Successfully uninstalled flatbuffers-23.5.26
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.12.0
        Uninstalling tensorflow-estimator-2.12.0:
          Successfully uninstalled tensorflow-estimator-2.12.0
      Attempting uninstall: tensorboard-data-server
        Found existing installation: tensorboard-data-server 0.7.0
        Uninstalling tensorboard-data-server-0.7.0:
          Successfully uninstalled tensorboard-data-server-0.7.0
      Attempting uninstall: protobuf
        Found existing installation: protobuf 4.23.2
        Uninstalling protobuf-4.23.2:
          Successfully uninstalled protobuf-4.23.2
      Attempting uninstall: google-auth-oauthlib
        Found existing installation: google-auth-oauthlib 1.0.0
        Uninstalling google-auth-oauthlib-1.0.0:
          Successfully uninstalled google-auth-oauthlib-1.0.0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.12.3
        Uninstalling tensorboard-2.12.3:
          Successfully uninstalled tensorboard-2.12.3
    Successfully installed flatbuffers-1.12 google-auth-oauthlib-0.4.6 keras-2.9.0 protobuf-3.19.6 tensorboard-2.9.1 tensorboard-data-server-0.6.1 tensorflow-estimator-2.9.0
    


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Input In [3], in <cell line: 2>()
          1 get_ipython().run_line_magic('pip', 'install tensorflow')
    ----> 2 import tensorflow as tf
          3 from tf.keras.preprocessing.text import Tokenizer
          4 from tf.keras.preprocessing.sequence import pad_sequences
    

    File ~\anaconda3\lib\site-packages\tensorflow\__init__.py:37, in <module>
         34 import sys as _sys
         35 import typing as _typing
    ---> 37 from tensorflow.python.tools import module_util as _module_util
         38 from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
         40 # Make sure code inside the TensorFlow codebase can use tf2.enabled() at import.
    

    File ~\anaconda3\lib\site-packages\tensorflow\python\__init__.py:37, in <module>
         29 # We aim to keep this file minimal and ideally remove completely.
         30 # If you are adding a new file with @tf_export decorators,
         31 # import it in modules_with_exports.py instead.
         32 
         33 # go/tf-wildcard-import
         34 # pylint: disable=wildcard-import,g-bad-import-order,g-import-not-at-top
         36 from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
    ---> 37 from tensorflow.python.eager import context
         39 # pylint: enable=wildcard-import
         40 
         41 # Bring in subpackages.
         42 from tensorflow.python import data
    

    File ~\anaconda3\lib\site-packages\tensorflow\python\eager\context.py:29, in <module>
         26 import numpy as np
         27 import six
    ---> 29 from tensorflow.core.framework import function_pb2
         30 from tensorflow.core.protobuf import config_pb2
         31 from tensorflow.core.protobuf import coordination_config_pb2
    

    File ~\anaconda3\lib\site-packages\tensorflow\core\framework\function_pb2.py:5, in <module>
          1 # -*- coding: utf-8 -*-
          2 # Generated by the protocol buffer compiler.  DO NOT EDIT!
          3 # source: tensorflow/core/framework/function.proto
          4 """Generated protocol buffer code."""
    ----> 5 from google.protobuf.internal import builder as _builder
          6 from google.protobuf import descriptor as _descriptor
          7 from google.protobuf import descriptor_pool as _descriptor_pool
    

    ImportError: cannot import name 'builder' from 'google.protobuf.internal' (C:\Users\LENOVO\anaconda3\lib\site-packages\google\protobuf\internal\__init__.py)



```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
print(stopwords.words('english'))
nltk.download('punkt')
nltk.download('omw-1.4')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\LENOVO\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\LENOVO\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\LENOVO\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to
    [nltk_data]     C:\Users\LENOVO\AppData\Roaming\nltk_data...
    

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    

    [nltk_data]   Package omw-1.4 is already up-to-date!
    




    True



# Data Preparing and Preprocessing


```python
from urllib import request

url1 = "https://www.gutenberg.org/cache/epub/16389/pg16389.txt"  #The Enchanted April          
url2 = "https://www.gutenberg.org/cache/epub/145/pg145.txt"      #Middlemarch     
url3 = "https://www.gutenberg.org/cache/epub/1259/pg1259.txt"    #Twenty Years After  
url4 = "https://www.gutenberg.org/files/1400/1400-0.txt"         #Great Expectations        
url5 = "https://www.gutenberg.org/cache/epub/11/pg11.txt"        #Alice's Adventures  

urls = [url1 ,url2 ,url3, url4 ,url5]
Books_names = ["The Enchanted April" ,"Middlemarch" , "Twenty Years After"  , "Great Expectations" ,"Alice's Adventures"]
Books_labels = [ 'a' ,'b' ,'c','d' ,'e']
Books_author = ["Elizabeth Von Arnim" ,"George Eliot" ,"Alexandre Dumas" , "Charles Dickens" ,"Lewis Carroll"]
Books_contents = []

for url  in urls :
    response = request.urlopen(url)
    raw = response.read().decode('utf8' )
    text= re.findall(r"[a-zA-Z]{3,}", raw)
    lemmatizer = WordNetLemmatizer()
    lst =[]
    for i in text:
        Word = i.lower()
        word = lemmatizer.lemmatize(Word)
        if word not in set(stopwords.words('english')):
            lst.append(str(word))
    Books_contents.append(lst)

Books_contents_100 = []               
for i in Books_contents:
    line = i[0:(math.floor(len(i)/100)) * 100]
    Books_contents_100.append(line)


data = pd.DataFrame()

for i in range(len(Books_contents_100)):
    dic = {}
    lst_100 =  [Books_contents_100[i][x:x+100] for x in range(0, len(Books_contents_100[i]), 100)]

    dic['index']= i  
    dic['Author']= Books_author[i]
    dic['title']= Books_names[i] 
    dic['label'] = Books_labels[i]
    dic['100_Words'] = lst_100 
    df_100 = pd.DataFrame(dic)
    for i in range(len(df_100)): 
        df_100["100_Words"][i] = " ".join(df_100["100_Words"][i])
    df = df_100[:200]
    data = data.append(df)
    data = shuffle(data)

data
```

    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:47: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_100["100_Words"][i] = " ".join(df_100["100_Words"][i])
    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:49: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      data = data.append(df)
    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:47: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_100["100_Words"][i] = " ".join(df_100["100_Words"][i])
    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:49: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      data = data.append(df)
    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:47: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_100["100_Words"][i] = " ".join(df_100["100_Words"][i])
    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:49: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      data = data.append(df)
    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:47: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_100["100_Words"][i] = " ".join(df_100["100_Words"][i])
    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13624\2244612486.py:49: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      data = data.append(df)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Author</th>
      <th>title</th>
      <th>label</th>
      <th>100_Words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>George Eliot</td>
      <td>Middlemarch</td>
      <td>b</td>
      <td>smaller brother seek martyrdom country moor to...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1</td>
      <td>George Eliot</td>
      <td>Middlemarch</td>
      <td>b</td>
      <td>would seeing much lively man shall inform reme...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>3</td>
      <td>Charles Dickens</td>
      <td>Great Expectations</td>
      <td>d</td>
      <td>grateful moral mystery seemed much company hub...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>3</td>
      <td>Charles Dickens</td>
      <td>Great Expectations</td>
      <td>d</td>
      <td>sent straight bed attic sloping roof wa low co...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0</td>
      <td>Elizabeth Von Arnim</td>
      <td>The Enchanted April</td>
      <td>a</td>
      <td>black sky san salvatore time comfortingly enco...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>4</td>
      <td>Lewis Carroll</td>
      <td>Alice's Adventures</td>
      <td>e</td>
      <td>small donation particularly important maintain...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2</td>
      <td>Alexandre Dumas</td>
      <td>Twenty Years After</td>
      <td>c</td>
      <td>away ambassador wa far away beyond sea illustr...</td>
    </tr>
    <tr>
      <th>189</th>
      <td>1</td>
      <td>George Eliot</td>
      <td>Middlemarch</td>
      <td>b</td>
      <td>would fully accounted perpetual crape poor add...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>2</td>
      <td>Alexandre Dumas</td>
      <td>Twenty Years After</td>
      <td>c</td>
      <td>minute planchet returned sir said one window c...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>Elizabeth Von Arnim</td>
      <td>The Enchanted April</td>
      <td>a</td>
      <td>nothing allowed mr wilkins leap conclusion unc...</td>
    </tr>
  </tbody>
</table>
<p>942 rows × 5 columns</p>
</div>



# Data Visualizations


```python
%pip install --upgrade pip
%pip install --upgrade Pillow
```

    Requirement already satisfied: pip in c:\users\lenovo\anaconda3\lib\site-packages (23.1.2)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    

    Requirement already satisfied: Pillow in c:\users\lenovo\anaconda3\lib\site-packages (9.5.0)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    

#Initialize a t-SNE object with the desired hyperparameters
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)

#Fit the t-SNE model to the data and obtain the low-dimensional representation
data_tsne = tsne.fit_transform(data)

#Visualize the low-dimensional representation using a scatter plot
plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
plt.show()


```python
import wordcloud

for label in data['label'].unique(): 
    books = data[data["label"]==label]["100_Words"]
    print(f"\n The most frequent 50 words of book {label}\n")
    wcloud = wordcloud.WordCloud(background_color='white', max_words=50, max_font_size=30)
    wcloud = wcloud.generate(str(books))
    plt.axis('off')
    plt.imshow(wcloud , cmap=None)
    plt.show()
```

    
     The most frequent 50 words of book b
    
    


    
![png](README_files/README_10_1.png)
    


    
     The most frequent 50 words of book d
    
    


    
![png](README_files/README_10_3.png)
    


    
     The most frequent 50 words of book a
    
    


    
![png](README_files/README_10_5.png)
    


    
     The most frequent 50 words of book c
    
    


    
![png](README_files/README_10_7.png)
    


    
     The most frequent 50 words of book e
    
    


    
![png](README_files/README_10_9.png)
    



```python
plt.figure(figsize=(10, 10))
sns.countplot(x=data['Author'])
plt.title("Count of Authors")
plt.show()
```


    
![png](README_files/README_11_0.png)
    


# Feature Engineering

###  1) Bag of Words(BOW)


```python
count_Vector= CountVectorizer()
bow = count_Vector.fit_transform(data['100_Words'])
bow_df = pd.DataFrame(bow.toarray(), columns=count_Vector.get_feature_names())
bow_df
```

    C:\Users\LENOVO\anaconda3\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aback</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abandonment</th>
      <th>abb</th>
      <th>abbey</th>
      <th>abdicated</th>
      <th>abear</th>
      <th>abeyance</th>
      <th>abhorrence</th>
      <th>...</th>
      <th>youngster</th>
      <th>youth</th>
      <th>youthful</th>
      <th>youthfulness</th>
      <th>zeal</th>
      <th>zealand</th>
      <th>zealous</th>
      <th>zigzag</th>
      <th>zip</th>
      <th>zounds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>937</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>938</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>939</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>940</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>941</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>942 rows × 10892 columns</p>
</div>



# Split the Data into Training, Validation and Testing


```python
X = bow.toarray()
Y = data['label']

print(X.shape)
print(Y.shape)
```

    (942, 10892)
    (942,)
    


```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = True, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state= 0)  
```


```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5)):
  
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, '-', color="g", label="Cross-validation score")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left');
    return plt
```


```python
def build_model(model, model_name, x_train, x_test, x_val, y_train, y_test, y_val, cross_valid):

    global general_model
    general_model = model.fit(x_train, y_train) 
    global y_pred
    y_pred = model.predict(x_test)
    global y_val_pred
    y_val_pred = model.predict(x_val)
    
    #K_Fold Cross Vaildation
    cv_accuracies = cross_val_score(estimator=model , X=x_train , y=y_train ,cv=cross_valid)
    #Measure the accuracy of the model (bais)
    accuracy_avg = cv_accuracies.mean()
    test_accuracy = accuracy_score(y_test, y_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    #Model performance evaluation 
    print("Cross Validation Accuracy :  ", cv_accuracies)
    print("\n Average Cross Validation Accuracy :  " , accuracy_avg)
    print("\n Testing accuracy :  "  ,test_accuracy)
    print("\n Validation accuracy :  "  ,val_accuracy)
    print("----------------------------------------------------------------------------------------------")

    print(classification_report(y_test, y_pred))

    #Confusion Matrix
    print('\n Confusion Matrix: \n')
    print(confusion_matrix(y_test, y_pred))
    plot_learning_curve(general_model,"Learning Curve of "+ model_name, x_train, y_train, ylim=(0, 1.1), cv=cross_valid, n_jobs=8)

    print("----------------------------------------------------------------------------------------------")
```


```python
def analyze_error(x_train, y_train, x_test, y_test ,y_pred):
    errors  = [] 
    errors_correct = []
    predict = []
    arr = np.array(y_test)

    for index ,text in enumerate(x_test) :
        if y_pred[index] != arr[index]:
            error = text
            errors.append(error)
            correct = arr[index]
            errors_correct.append(correct)
            pred = y_pred[index]
            predict.append(pred)
            
    document_df = pd.DataFrame()
    document_df['Document_error'] = errors
    document_df['Correct']   = correct
    document_df['Predicted'] = predict
 
    print(" The Documents misclassified by the model are : " , len(errors))
    print("----------------------------------------------------------------------------------------------\n")
 
    label_encoder = preprocessing.LabelEncoder()
    x_train_cp = np.copy(x_train)
    x_test_cp = np.copy(x_test)
    y_train_cp = np.copy(y_train)
    y_test_cp = np.copy(y_test)

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(general_model, np.array(x_train_cp), label_encoder.fit_transform(y_train_cp), np.array(x_test_cp), label_encoder.fit_transform(y_test_cp), num_rounds=2, random_seed=0)
  
    print('Average bias : %.3f' % avg_bias)
    print('Average variance : %.3f' % avg_var)
    print("----------------------------------------------------------------------------------------------")

    return document_df 
```

# Support Vector Machine (BOW)


```python
svm_bow = SVC(kernel='linear')
build_model(svm_bow, "SVM with BOW", X_train, X_test, X_val, y_train, y_test ,y_val, 10)

svm_bow_error = analyze_error(X_train, y_train, X_test, y_test, y_pred)
display(svm_bow_error.head())
```

    Cross Validation Accuracy :   [0.96491228 0.98245614 0.98245614 0.92982456 0.98214286 0.96428571
     0.98214286 0.96428571 0.98214286 0.98214286]
    
     Average Cross Validation Accuracy :   0.9716791979949875
    
     Testing accuracy :   0.9788359788359788
    
     Validation accuracy :   0.9894179894179894
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.98      0.99        46
               b       0.95      1.00      0.97        37
               c       1.00      0.97      0.99        40
               d       0.95      0.98      0.96        42
               e       1.00      0.96      0.98        24
    
        accuracy                           0.98       189
       macro avg       0.98      0.98      0.98       189
    weighted avg       0.98      0.98      0.98       189
    
    
     Confusion Matrix: 
    
    [[45  0  0  1  0]
     [ 0 37  0  0  0]
     [ 0  0 39  1  0]
     [ 0  1  0 41  0]
     [ 0  1  0  0 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  4
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.032
    Average variance : 0.008
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>d</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_22_2.png)
    


# Random Forest (BOW)


```python
rf_bow = RandomForestClassifier(n_estimators=100)
build_model(rf_bow, "Random Forest with BOW", X_train, X_test, X_val, y_train, y_test ,y_val ,10)

rf_bow_error = analyze_error(X_train, y_train, X_test, y_test, y_pred)
display(rf_bow_error.head())
```

    Cross Validation Accuracy :   [0.98245614 1.         1.         0.98245614 0.98214286 0.96428571
     1.         0.98214286 1.         0.98214286]
    
     Average Cross Validation Accuracy :   0.9875626566416041
    
     Testing accuracy :   0.9523809523809523
    
     Validation accuracy :   0.9894179894179894
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       0.94      0.96      0.95        46
               b       0.92      0.97      0.95        37
               c       1.00      0.97      0.99        40
               d       0.93      0.93      0.93        42
               e       1.00      0.92      0.96        24
    
        accuracy                           0.95       189
       macro avg       0.96      0.95      0.95       189
    weighted avg       0.95      0.95      0.95       189
    
    
     Confusion Matrix: 
    
    [[44  0  0  2  0]
     [ 1 36  0  0  0]
     [ 0  0 39  1  0]
     [ 2  1  0 39  0]
     [ 0  2  0  0 22]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  9
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.048
    Average variance : 0.024
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_24_2.png)
    


# Naïve Bayes (BOW)


```python
nb_bow = MultinomialNB()
build_model(nb_bow, "Naïve Bayes with BOW", X_train, X_test, X_val, y_train, y_test ,y_val ,10)

nb_bow_error = analyze_error(X_train, y_train, X_test, y_test, y_pred)
display(nb_bow_error.head())
```

    Cross Validation Accuracy :   [0.98245614 1.         0.98245614 0.98245614 1.         1.
     0.98214286 0.98214286 1.         1.        ]
    
     Average Cross Validation Accuracy :   0.9911654135338346
    
     Testing accuracy :   0.9841269841269841
    
     Validation accuracy :   0.9947089947089947
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.98      0.99        46
               b       1.00      1.00      1.00        37
               c       0.97      0.97      0.97        40
               d       1.00      0.98      0.99        42
               e       0.92      1.00      0.96        24
    
        accuracy                           0.98       189
       macro avg       0.98      0.99      0.98       189
    weighted avg       0.98      0.98      0.98       189
    
    
     Confusion Matrix: 
    
    [[45  0  0  0  1]
     [ 0 37  0  0  0]
     [ 0  0 39  0  1]
     [ 0  0  1 41  0]
     [ 0  0  0  0 24]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  3
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.026
    Average variance : 0.013
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_26_2.png)
    


# K-Nearest Neighbour (BOW)


```python
knn_bow = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', p=1)
build_model(knn_bow, "KNN with BOW", X_train, X_test, X_val, y_train, y_test ,y_val ,10)

KNN_Bow_error = analyze_error(X_train, y_train, X_test, y_test ,y_pred)
display(KNN_Bow_error.head())
```

    Cross Validation Accuracy :   [0.92982456 0.96491228 0.98245614 0.9122807  0.92857143 0.92857143
     0.94642857 0.94642857 0.92857143 0.875     ]
    
     Average Cross Validation Accuracy :   0.9343045112781955
    
     Testing accuracy :   0.8941798941798942
    
     Validation accuracy :   0.9365079365079365
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       0.78      0.93      0.85        46
               b       0.94      0.92      0.93        37
               c       0.97      0.93      0.95        40
               d       1.00      0.74      0.85        42
               e       0.83      1.00      0.91        24
    
        accuracy                           0.89       189
       macro avg       0.91      0.90      0.90       189
    weighted avg       0.91      0.89      0.89       189
    
    
     Confusion Matrix: 
    
    [[43  1  0  0  2]
     [ 3 34  0  0  0]
     [ 2  1 37  0  0]
     [ 7  0  1 31  3]
     [ 0  0  0  0 24]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  20
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.106
    Average variance : 0.056
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ...</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>a</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_28_2.png)
    


# XG-Boost (BOW)


```python
le = preprocessing.LabelEncoder()
xgb_bow = XGBClassifier()

y_train_XGBoost = le.fit_transform(y_train)
y_test_XGBoost = le.fit_transform(y_test)
y_val_XGBoost = le.fit_transform(y_val)

build_model(xgb_bow, "XG-Boost with BOW", X_train, X_test, X_val, y_train_XGBoost, y_test_XGBoost ,y_val_XGBoost ,10)

xgb_bow_error = analyze_error(X_train, y_train, X_test, y_test, y_pred)
display(xgb_bow_error.head())
```

    Cross Validation Accuracy :   [0.94736842 0.96491228 0.96491228 0.94736842 0.96428571 0.92857143
     0.98214286 0.96428571 0.94642857 0.89285714]
    
     Average Cross Validation Accuracy :   0.9503132832080199
    
     Testing accuracy :   0.9470899470899471
    
     Validation accuracy :   0.9576719576719577
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.98      0.89      0.93        46
               1       0.93      1.00      0.96        37
               2       1.00      0.95      0.97        40
               3       0.87      0.95      0.91        42
               4       1.00      0.96      0.98        24
    
        accuracy                           0.95       189
       macro avg       0.95      0.95      0.95       189
    weighted avg       0.95      0.95      0.95       189
    
    
     Confusion Matrix: 
    
    [[41  0  0  5  0]
     [ 0 37  0  0  0]
     [ 1  1 38  0  0]
     [ 0  2  0 40  0]
     [ 0  0  0  1 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  189
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.069
    Average variance : 0.029
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_30_2.png)
    


# Stochastic Gradient Descent (BOW)


```python
sgd_bow = SGDClassifier(loss='log')
build_model(sgd_bow, "Stochastic Gradient Descent with BOW", X_train, X_test, X_val, y_train, y_test ,y_val ,10)

sgd_bow_error = analyze_error(X_train, y_train, X_test, y_test, y_pred)
display(sgd_bow_error.head())
```

    Cross Validation Accuracy :   [0.94736842 0.94736842 1.         0.94736842 0.98214286 0.96428571
     0.94642857 0.96428571 0.98214286 0.98214286]
    
     Average Cross Validation Accuracy :   0.9663533834586466
    
     Testing accuracy :   0.9735449735449735
    
     Validation accuracy :   0.9947089947089947
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.96      0.98        46
               b       0.95      1.00      0.97        37
               c       0.97      0.97      0.97        40
               d       0.95      0.98      0.96        42
               e       1.00      0.96      0.98        24
    
        accuracy                           0.97       189
       macro avg       0.98      0.97      0.97       189
    weighted avg       0.97      0.97      0.97       189
    
    
     Confusion Matrix: 
    
    [[44  1  1  0  0]
     [ 0 37  0  0  0]
     [ 0  0 39  1  0]
     [ 0  1  0 41  0]
     [ 0  0  0  1 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  5
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.048
    Average variance : 0.034
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>d</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>c</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_32_2.png)
    


###  2) Term Frequency - Inverse Document Frequency (TF_IDF)


```python
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(data['100_Words'])
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
tfidf_df
```

    C:\Users\LENOVO\anaconda3\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aback</th>
      <th>abandon</th>
      <th>abandoned</th>
      <th>abandonment</th>
      <th>abb</th>
      <th>abbey</th>
      <th>abdicated</th>
      <th>abear</th>
      <th>abeyance</th>
      <th>abhorrence</th>
      <th>...</th>
      <th>youngster</th>
      <th>youth</th>
      <th>youthful</th>
      <th>youthfulness</th>
      <th>zeal</th>
      <th>zealand</th>
      <th>zealous</th>
      <th>zigzag</th>
      <th>zip</th>
      <th>zounds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.105791</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>937</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>938</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>939</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>940</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>941</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>942 rows × 10892 columns</p>
</div>



# Split the Data into Training, Validation and Testing


```python
X_tfidf = tfidf.toarray()
y_tfidf = data['label']

print(X_tfidf.shape)
print(y_tfidf.shape)
```

    (942, 10892)
    (942,)
    


```python
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_tfidf, test_size=0.2, shuffle = True, random_state = 0)
X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf = train_test_split(X_train_tfidf, y_train_tfidf, test_size=0.25, random_state= 0)  
```

# Support Vector Machine (TF_IDF)


```python
svm_tfidf = svm.SVC(kernel='sigmoid')
build_model(svm_tfidf, "Support Vector Machine with TFIDF", X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train_tfidf, y_test_tfidf ,y_val_tfidf ,10)

SVM_tfidf_error  = analyze_error(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf ,y_pred)
display(SVM_tfidf_error.head())
```

    Cross Validation Accuracy :   [0.98245614 0.98245614 0.98245614 0.94736842 1.         0.98214286
     0.98214286 0.98214286 0.98214286 1.        ]
    
     Average Cross Validation Accuracy :   0.982330827067669
    
     Testing accuracy :   0.9788359788359788
    
     Validation accuracy :   0.9947089947089947
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.98      0.99        46
               b       0.97      1.00      0.99        37
               c       0.95      0.97      0.96        40
               d       1.00      0.98      0.99        42
               e       0.96      0.96      0.96        24
    
        accuracy                           0.98       189
       macro avg       0.98      0.98      0.98       189
    weighted avg       0.98      0.98      0.98       189
    
    
     Confusion Matrix: 
    
    [[45  0  1  0  0]
     [ 0 37  0  0  0]
     [ 0  0 39  0  1]
     [ 0  0  1 41  0]
     [ 0  1  0  0 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  4
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.026
    Average variance : 0.011
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_39_2.png)
    


# Random Forest (TF_IDF)


```python
rf_tfidf = RandomForestClassifier(n_estimators=100)
build_model(rf_tfidf, "Random Forest with TFIDF", X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train_tfidf, y_test_tfidf ,y_val_tfidf ,10)

rf_tfidf_error = analyze_error(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, y_pred)
display(rf_tfidf_error.head())
```

    Cross Validation Accuracy :   [0.98245614 1.         0.98245614 0.94736842 1.         0.96428571
     1.         0.96428571 0.98214286 1.        ]
    
     Average Cross Validation Accuracy :   0.9822994987468672
    
     Testing accuracy :   0.9682539682539683
    
     Validation accuracy :   0.9788359788359788
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       0.98      0.96      0.97        46
               b       0.93      1.00      0.96        37
               c       1.00      1.00      1.00        40
               d       0.95      0.95      0.95        42
               e       1.00      0.92      0.96        24
    
        accuracy                           0.97       189
       macro avg       0.97      0.97      0.97       189
    weighted avg       0.97      0.97      0.97       189
    
    
     Confusion Matrix: 
    
    [[44  0  0  2  0]
     [ 0 37  0  0  0]
     [ 0  0 40  0  0]
     [ 1  1  0 40  0]
     [ 0  2  0  0 22]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  6
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.063
    Average variance : 0.032
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>a</td>
      <td>d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_41_2.png)
    


# Naïve Bayes (TF_IDF)


```python
nb_tfidf = MultinomialNB()
build_model(nb_tfidf, "Naïve Bayes with TFIDF", X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train_tfidf, y_test_tfidf ,y_val_tfidf ,10)

nb_tfidf_error = analyze_error(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, y_pred)
display(nb_tfidf_error.head())
```

    Cross Validation Accuracy :   [0.98245614 1.         0.96491228 0.92982456 0.98214286 0.98214286
     0.98214286 0.96428571 0.98214286 1.        ]
    
     Average Cross Validation Accuracy :   0.9770050125313283
    
     Testing accuracy :   0.9735449735449735
    
     Validation accuracy :   0.9894179894179894
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       0.98      0.98      0.98        46
               b       0.97      1.00      0.99        37
               c       0.95      0.97      0.96        40
               d       1.00      0.95      0.98        42
               e       0.96      0.96      0.96        24
    
        accuracy                           0.97       189
       macro avg       0.97      0.97      0.97       189
    weighted avg       0.97      0.97      0.97       189
    
    
     Confusion Matrix: 
    
    [[45  0  1  0  0]
     [ 0 37  0  0  0]
     [ 0  0 39  0  1]
     [ 1  0  1 40  0]
     [ 0  1  0  0 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  5
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.132
    Average variance : 0.066
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>c</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_43_2.png)
    


# K-Nearest Neighbour (TF_IDF)


```python
knn_tfidf =  KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', p=2)
build_model(knn_tfidf, "K-Nearest Neighbour with TFIDF", X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train_tfidf, y_test_tfidf ,y_val_tfidf ,10)

knn_tfidf_error = analyze_error(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf ,y_pred)
display(knn_tfidf_error.head())
```

    Cross Validation Accuracy :   [0.94736842 0.96491228 0.98245614 0.9122807  0.94642857 0.96428571
     0.94642857 0.94642857 0.96428571 0.98214286]
    
     Average Cross Validation Accuracy :   0.9557017543859649
    
     Testing accuracy :   0.9576719576719577
    
     Validation accuracy :   0.9682539682539683
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       0.92      0.98      0.95        46
               b       0.95      0.97      0.96        37
               c       0.97      0.95      0.96        40
               d       1.00      0.90      0.95        42
               e       0.96      1.00      0.98        24
    
        accuracy                           0.96       189
       macro avg       0.96      0.96      0.96       189
    weighted avg       0.96      0.96      0.96       189
    
    
     Confusion Matrix: 
    
    [[45  1  0  0  0]
     [ 1 36  0  0  0]
     [ 0  1 38  0  1]
     [ 3  0  1 38  0]
     [ 0  0  0  0 24]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  8
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.058
    Average variance : 0.029
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>e</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_45_2.png)
    


# XG-Boost (TF_IDF)


```python
le = preprocessing.LabelEncoder()
xgb_tfidf = XGBClassifier()

y_train_tfidf_XGBoost = le.fit_transform(y_train_tfidf)
y_test_tfidf_XGBoost = le.fit_transform(y_test_tfidf)
y_val_tfidf_XGBoost = le.fit_transform(y_val_tfidf)

build_model(xgb_tfidf, "XG-Boost with TFIDF", X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train_tfidf_XGBoost, y_test_tfidf_XGBoost ,y_val_tfidf_XGBoost ,10)

xgb_tfidf_error = analyze_error(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, y_pred)
display(xgb_tfidf_error.head())
```

    Cross Validation Accuracy :   [0.96491228 0.96491228 0.96491228 0.92982456 0.92857143 0.92857143
     0.96428571 0.98214286 0.94642857 0.91071429]
    
     Average Cross Validation Accuracy :   0.9485275689223058
    
     Testing accuracy :   0.9365079365079365
    
     Validation accuracy :   0.9788359788359788
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.95      0.91      0.93        46
               1       0.88      1.00      0.94        37
               2       1.00      0.93      0.96        40
               3       0.93      0.93      0.93        42
               4       0.92      0.92      0.92        24
    
        accuracy                           0.94       189
       macro avg       0.94      0.94      0.94       189
    weighted avg       0.94      0.94      0.94       189
    
    
     Confusion Matrix: 
    
    [[42  0  0  2  2]
     [ 0 37  0  0  0]
     [ 1  2 37  0  0]
     [ 1  2  0 39  0]
     [ 0  1  0  1 22]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  189
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.079
    Average variance : 0.029
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>b</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_47_2.png)
    


# Stochastic Gradient Descent (TF_IDF)


```python
sgd_tfidf = SGDClassifier(loss='log')
build_model(sgd_tfidf, "Stochastic Gradient Descent with TFIDF", X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train_tfidf, y_test_tfidf ,y_val_tfidf ,10)

sgd_tfidf_error = analyze_error(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, y_pred)
display(sgd_tfidf_error.head())
```

    Cross Validation Accuracy :   [0.96491228 0.98245614 0.98245614 0.94736842 1.         0.96428571
     1.         0.98214286 0.98214286 1.        ]
    
     Average Cross Validation Accuracy :   0.9805764411027569
    
     Testing accuracy :   0.9788359788359788
    
     Validation accuracy :   0.9947089947089947
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.98      0.99        46
               b       0.93      1.00      0.96        37
               c       1.00      0.97      0.99        40
               d       1.00      0.98      0.99        42
               e       0.96      0.96      0.96        24
    
        accuracy                           0.98       189
       macro avg       0.98      0.98      0.98       189
    weighted avg       0.98      0.98      0.98       189
    
    
     Confusion Matrix: 
    
    [[45  1  0  0  0]
     [ 0 37  0  0  0]
     [ 0  0 39  0  1]
     [ 0  1  0 41  0]
     [ 0  1  0  0 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  4
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.026
    Average variance : 0.003
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_49_2.png)
    


###  3) N-Gram


```python
bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
bigram = bigram_vectorizer.fit_transform(data['100_Words'])
bigram_df = pd.DataFrame(bigram.toarray(), columns= bigram_vectorizer.get_feature_names())
bigram_df
```

    C:\Users\LENOVO\anaconda3\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aback</th>
      <th>aback constant</th>
      <th>aback mr</th>
      <th>aback said</th>
      <th>abandon</th>
      <th>abandon mazarin</th>
      <th>abandon stream</th>
      <th>abandon thus</th>
      <th>abandoned</th>
      <th>abandoned every</th>
      <th>...</th>
      <th>zigzag arrive</th>
      <th>zigzag path</th>
      <th>zigzag wa</th>
      <th>zip</th>
      <th>zip associated</th>
      <th>zounds</th>
      <th>zounds cried</th>
      <th>zounds flavor</th>
      <th>zounds said</th>
      <th>zounds weapon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>937</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>938</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>939</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>940</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>941</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>942 rows × 86554 columns</p>
</div>



# Split the Data into Training, Validation and Testing


```python
X_ngram = bigram.toarray()
y_ngram = data['label']

print(X_ngram.shape)
print(y_ngram.shape)
```

    (942, 86554)
    (942,)
    


```python
X_train_ngram, X_test_ngram, y_train_ngram, y_test_ngram = train_test_split(X_ngram, y_ngram, test_size=0.2, shuffle = True, random_state = 0)
X_train_ngram, X_val_ngram, y_train_ngram, y_val_ngram = train_test_split(X_train_ngram, y_train_ngram, test_size=0.25, random_state= 0)  
```

# Support Vector Machine (N-Gram)


```python
svm_n_gram = svm.SVC(kernel='sigmoid')
build_model(svm_n_gram, "Support Vector Machine with N-gram", X_train_ngram, X_test_ngram, X_val_ngram, y_train_ngram, y_test_ngram, y_val_ngram ,10)

svm_ngram_error = analyze_error(X_train_ngram, y_train_ngram, X_test_ngram, y_test_ngram ,y_pred)
display(svm_ngram_error.head())
```

    Cross Validation Accuracy :   [0.96491228 0.98245614 0.98245614 0.9122807  0.94642857 0.96428571
     1.         0.98214286 0.98214286 0.98214286]
    
     Average Cross Validation Accuracy :   0.9699248120300753
    
     Testing accuracy :   0.9682539682539683
    
     Validation accuracy :   0.9894179894179894
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.93      0.97        46
               b       0.90      1.00      0.95        37
               c       0.95      0.97      0.96        40
               d       1.00      0.98      0.99        42
               e       1.00      0.96      0.98        24
    
        accuracy                           0.97       189
       macro avg       0.97      0.97      0.97       189
    weighted avg       0.97      0.97      0.97       189
    
    
     Confusion Matrix: 
    
    [[43  2  1  0  0]
     [ 0 37  0  0  0]
     [ 0  1 39  0  0]
     [ 0  0  1 41  0]
     [ 0  1  0  0 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  6
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.032
    Average variance : 0.013
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>c</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_56_2.png)
    


# Random Forest (N-Gram)


```python
rf_ngram = RandomForestClassifier(n_estimators=100)
build_model(rf_ngram, "Random Forest with N-gram", X_train_ngram, X_test_ngram, X_val_ngram, y_train_ngram, y_test_ngram, y_val_ngram ,10)

rf_ngram_error = analyze_error(X_train_ngram, y_train_ngram, X_test_ngram, y_test_ngram, y_pred)
display(rf_ngram_error.head())
```

    Cross Validation Accuracy :   [0.98245614 1.         0.98245614 0.96491228 1.         0.96428571
     1.         0.98214286 1.         1.        ]
    
     Average Cross Validation Accuracy :   0.987625313283208
    
     Testing accuracy :   0.9682539682539683
    
     Validation accuracy :   0.9894179894179894
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.96      0.98        46
               b       0.88      1.00      0.94        37
               c       1.00      1.00      1.00        40
               d       1.00      0.95      0.98        42
               e       0.96      0.92      0.94        24
    
        accuracy                           0.97       189
       macro avg       0.97      0.97      0.97       189
    weighted avg       0.97      0.97      0.97       189
    
    
     Confusion Matrix: 
    
    [[44  1  0  0  1]
     [ 0 37  0  0  0]
     [ 0  0 40  0  0]
     [ 0  2  0 40  0]
     [ 0  2  0  0 22]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  6
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.063
    Average variance : 0.032
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>e</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>a</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_58_2.png)
    


# Naïve Bayes (N-Gram)


```python
nb_ngram = MultinomialNB()
build_model(nb_ngram, "Naïve Bayes with N-gram", X_train_ngram, X_test_ngram, X_val_ngram, y_train_ngram, y_test_ngram, y_val_ngram ,10)

nb_ngram_error = analyze_error(X_train_ngram, y_train_ngram, X_test_ngram, y_test_ngram, y_pred)
display(nb_ngram_error.head())
```

    Cross Validation Accuracy :   [0.98245614 1.         0.98245614 0.98245614 1.         1.
     0.98214286 0.98214286 1.         1.        ]
    
     Average Cross Validation Accuracy :   0.9911654135338346
    
     Testing accuracy :   0.9841269841269841
    
     Validation accuracy :   0.9947089947089947
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.98      0.99        46
               b       0.95      1.00      0.97        37
               c       1.00      0.97      0.99        40
               d       1.00      0.98      0.99        42
               e       0.96      1.00      0.98        24
    
        accuracy                           0.98       189
       macro avg       0.98      0.99      0.98       189
    weighted avg       0.98      0.98      0.98       189
    
    
     Confusion Matrix: 
    
    [[45  1  0  0  0]
     [ 0 37  0  0  0]
     [ 0  0 39  0  1]
     [ 0  1  0 41  0]
     [ 0  0  0  0 24]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  3
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.026
    Average variance : 0.008
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_60_2.png)
    


# K-Nearest Neighbour (N-Gram)


```python
knn_n_gram =  KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', p=2)
build_model(knn_n_gram, "K-Nearest Neighbour with N-Gram", X_train_ngram, X_test_ngram, X_val_ngram, y_train_ngram, y_test_ngram, y_val_ngram ,10)

knn_ngram_error = analyze_error(X_train_ngram, y_train_ngram, X_test_ngram, y_test_ngram, y_pred)
display(knn_ngram_error.head())
```

    Cross Validation Accuracy :   [0.92982456 0.92982456 0.96491228 0.9122807  0.83928571 0.91071429
     0.98214286 0.91071429 0.94642857 0.92857143]
    
     Average Cross Validation Accuracy :   0.9254699248120302
    
     Testing accuracy :   0.8941798941798942
    
     Validation accuracy :   0.9206349206349206
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       0.98      0.93      0.96        46
               b       0.71      0.97      0.82        37
               c       0.95      0.93      0.94        40
               d       0.97      0.74      0.84        42
               e       0.96      0.92      0.94        24
    
        accuracy                           0.89       189
       macro avg       0.91      0.90      0.90       189
    weighted avg       0.91      0.89      0.90       189
    
    
     Confusion Matrix: 
    
    [[43  2  1  0  0]
     [ 0 36  0  1  0]
     [ 0  2 37  0  1]
     [ 1  9  1 31  0]
     [ 0  2  0  0 22]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  20
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.175
    Average variance : 0.079
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>d</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_62_2.png)
    


# XG-Boost (N-Gram)


```python
le = preprocessing.LabelEncoder()
xgb_ngram = XGBClassifier()

y_train_ngram_XGBoost = le.fit_transform(y_train_ngram)
y_test_ngram_XGBoost = le.fit_transform(y_test_ngram)
y_val_ngram_XGBoost = le.fit_transform(y_val_ngram)

build_model(xgb_ngram, "XG-Boost with N-Gram", X_train_ngram, X_test_ngram, X_val_ngram, y_train_ngram_XGBoost, y_test_ngram_XGBoost, y_val_ngram_XGBoost ,10)

xgb_ngram_error = analyze_error(X_train_ngram, y_train_ngram, X_test_ngram, y_test_ngram, y_pred)
display(xgb_ngram_error.head())
```

    Cross Validation Accuracy :   [0.94736842 0.96491228 0.96491228 0.94736842 0.92857143 0.94642857
     0.98214286 0.94642857 0.94642857 0.91071429]
    
     Average Cross Validation Accuracy :   0.9485275689223058
    
     Testing accuracy :   0.9470899470899471
    
     Validation accuracy :   0.9629629629629629
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.95      0.89      0.92        46
               1       0.93      1.00      0.96        37
               2       1.00      0.95      0.97        40
               3       0.91      0.95      0.93        42
               4       0.96      0.96      0.96        24
    
        accuracy                           0.95       189
       macro avg       0.95      0.95      0.95       189
    weighted avg       0.95      0.95      0.95       189
    
    
     Confusion Matrix: 
    
    [[41  1  0  3  1]
     [ 0 37  0  0  0]
     [ 1  1 38  0  0]
     [ 1  1  0 40  0]
     [ 0  0  0  1 23]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  189
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.053
    Average variance : 0.034
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_64_2.png)
    


# Stochastic Gradient Descent (N-Gram)


```python
#The Champion Model
sgd_ngram = SGDClassifier(loss='log')
build_model(sgd_ngram, "Stochastic Gradient Descent with N-Gram", X_train_ngram, X_test_ngram, X_val_ngram, y_train_ngram, y_test_ngram, y_val_ngram ,10)

sgd_ngram_error = analyze_error(X_train_ngram, y_train_ngram, X_test_ngram, y_test_ngram, y_pred)
display(sgd_ngram_error.head())
```

    Cross Validation Accuracy :   [0.96491228 0.92982456 0.98245614 0.92982456 1.         0.94642857
     1.         0.94642857 0.98214286 0.96428571]
    
     Average Cross Validation Accuracy :   0.9646303258145362
    
     Testing accuracy :   0.9788359788359788
    
     Validation accuracy :   0.9735449735449735
    ----------------------------------------------------------------------------------------------
                  precision    recall  f1-score   support
    
               a       1.00      0.98      0.99        46
               b       0.95      0.97      0.96        37
               c       1.00      0.97      0.99        40
               d       1.00      0.98      0.99        42
               e       0.92      1.00      0.96        24
    
        accuracy                           0.98       189
       macro avg       0.97      0.98      0.98       189
    weighted avg       0.98      0.98      0.98       189
    
    
     Confusion Matrix: 
    
    [[45  1  0  0  0]
     [ 0 36  0  0  1]
     [ 0  0 39  0  1]
     [ 0  1  0 41  0]
     [ 0  0  0  0 24]]
    ----------------------------------------------------------------------------------------------
     The Documents misclassified by the model are :  4
    ----------------------------------------------------------------------------------------------
    
    Average bias : 0.063
    Average variance : 0.037
    ----------------------------------------------------------------------------------------------
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Document_error</th>
      <th>Correct</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>b</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](README_files/README_66_2.png)
    


# Plotting Learning Curve 


```python
#BOW
plot_learning_curve(svm_bow, "SVM based on BOW", X_train, y_train, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("svm_bow.png")

plot_learning_curve(rf_bow, "Random Forest based on BOW", X_train, y_train, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("rf_bow.png")

plot_learning_curve(nb_bow, "Naïve Bayes based on BOW", X_train, y_train, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("nb_bow.png")

plot_learning_curve(knn_bow, "KNN based on BOW", X_train, y_train, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("knn_bow.png")

plot_learning_curve(xgb_bow, "XG Boost based on BOW", X_train, y_train_XGBoost, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("xgboost_bow.png")

plot_learning_curve(sgd_bow, "SGD based on BOW", X_train, y_train, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("sgd_bow.png")
```


    
![png](README_files/README_68_0.png)
    



    
![png](README_files/README_68_1.png)
    



    
![png](README_files/README_68_2.png)
    



    
![png](README_files/README_68_3.png)
    



    
![png](README_files/README_68_4.png)
    



    
![png](README_files/README_68_5.png)
    



```python
#TF_IDF
plot_learning_curve(svm_tfidf, "SVM based on TF_IDF", X_train_tfidf, y_train_tfidf, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("svm_tfidf.png")

plot_learning_curve(rf_tfidf, "Random Forest based on TF_IDF", X_train_tfidf, y_train_tfidf, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("rf_tfidf.png")

plot_learning_curve(nb_tfidf, "Naïve Bayes based on TF_IDF", X_train_tfidf, y_train_tfidf, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("nb_tfidf.png")

plot_learning_curve(knn_tfidf, "KNN based on TF_IDF", X_train_tfidf, y_train_tfidf, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("knn_tfidf.png")

plot_learning_curve(xgb_tfidf, "XG Boost based on TF_IDF", X_train_tfidf,  y_train_tfidf_XGBoost, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("xgboost_tfidf.png")

plot_learning_curve(sgd_tfidf, "SGD based on TF_IDF", X_train_tfidf, y_train_tfidf, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("sgd_tfidf.png")
```


    
![png](README_files/README_69_0.png)
    



    
![png](README_files/README_69_1.png)
    



    
![png](README_files/README_69_2.png)
    



    
![png](README_files/README_69_3.png)
    



    
![png](README_files/README_69_4.png)
    



    
![png](README_files/README_69_5.png)
    



```python
#N-Gram
plot_learning_curve(svm_n_gram, "SVM based on N-Gram", X_train_ngram, y_train_ngram, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("svm_ngram.png")

plot_learning_curve(rf_ngram, "Random Forest based on N-Gram", X_train_ngram, y_train_ngram, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("rf_ngram.png")

plot_learning_curve(nb_ngram, "Naïve Bayes based on N-Gram", X_train_ngram, y_train_ngram, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("nb_ngram.png")

plot_learning_curve(knn_n_gram, "KNN based on N-Gram", X_train_ngram, y_train_ngram, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("knn_ngram.png")

plot_learning_curve(xgb_ngram, "XG Boost based on N-Gram", X_train_ngram,  y_train_ngram_XGBoost, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("xgboost_ngram.png")

plot_learning_curve(sgd_ngram, "SGD based on N-Gram", X_train_ngram, y_train_ngram, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0,5))
plt.savefig("sgd_ngram.png")
```


    
![png](README_files/README_70_0.png)
    



    
![png](README_files/README_70_1.png)
    



    
![png](README_files/README_70_2.png)
    



    
![png](README_files/README_70_3.png)
    



    
![png](README_files/README_70_4.png)
    



    
![png](README_files/README_70_5.png)
    


# Text Classification using BERT


```python
%pip install transformers
```

    Requirement already satisfied: transformers in c:\users\lenovo\anaconda3\lib\site-packages (4.29.2)
    Requirement already satisfied: filelock in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (3.6.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.14.1 in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (0.15.1)
    Requirement already satisfied: numpy>=1.17 in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (1.22.4)
    Requirement already satisfied: packaging>=20.0 in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (21.3)
    Requirement already satisfied: pyyaml>=5.1 in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (6.0)
    Requirement already satisfied: regex!=2019.12.17 in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (2022.3.15)
    Requirement already satisfied: requests in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (2.27.1)Note: you may need to restart the kernel to use updated packages.
    
    

    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -0otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -1otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -2otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -0otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -1otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -2otobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -illow (c:\users\lenovo\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\lenovo\anaconda3\lib\site-packages)
    

    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (0.13.3)
    Requirement already satisfied: tqdm>=4.27 in c:\users\lenovo\anaconda3\lib\site-packages (from transformers) (4.64.0)
    Requirement already satisfied: fsspec in c:\users\lenovo\anaconda3\lib\site-packages (from huggingface-hub<1.0,>=0.14.1->transformers) (2022.2.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\users\lenovo\anaconda3\lib\site-packages (from huggingface-hub<1.0,>=0.14.1->transformers) (3.7.4.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\lenovo\anaconda3\lib\site-packages (from packaging>=20.0->transformers) (3.0.4)
    Requirement already satisfied: colorama in c:\users\lenovo\anaconda3\lib\site-packages (from tqdm>=4.27->transformers) (0.4.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\lenovo\anaconda3\lib\site-packages (from requests->transformers) (1.26.9)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\lenovo\anaconda3\lib\site-packages (from requests->transformers) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\lenovo\anaconda3\lib\site-packages (from requests->transformers) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\lenovo\anaconda3\lib\site-packages (from requests->transformers) (3.3)
    


```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
```


```python
# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
```


```python
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```


```python
# Tokenize the input data
def tokenize_data(text, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for sentence in text:
        encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

```


```python
train_inputs, train_masks = tokenize_data(train_data['100_Words'], tokenizer, 100)
val_inputs, val_masks = tokenize_data(val_data['100_Words'], tokenizer, 100)
```


```python
# Convert the labels to integers
label_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
train_labels = torch.tensor([label_dict[label] for label in train_data['label']])
val_labels = torch.tensor([label_dict[label] for label in val_data['label']])
```


```python
# Create a PyTorch dataset
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
```


```python
# Create a DataLoader to feed the data into the model in batches
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```


```python
# Load the pre-trained BERT model and modify the final layer to match the number of labels
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
```


```python
# Set the device to use (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```


```python
# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)
```


```python
# Train the model
epochs = 4
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.zero_grad()
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average training loss: {}".format(avg_train_loss))

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        total_eval_accuracy += (logits.argmax(axis=1) == label_ids).sum()
        nb_eval_steps += 1
    avg_val_accuracy = total_eval_accuracy / (len(val_dataloader) * batch_size)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    print("Validation accuracy: {}".format(avg_val_accuracy))
    print("Validation loss: {}".format(avg_val_loss))
```


```python
#Andrew Adel Labib
#Hussien Amin Abdelhafez
#Phoebe Thabit Wadea
#Sandy Adel Latef
```


```python
!jupyter nbconvert --to markdown 'Classification Assignment.ipynb' --output README.md
```

    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only 
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place, 
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document. 
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf']
    

    [NbConvertApp] WARNING | pattern "'Classification" matched no files
    [NbConvertApp] WARNING | pattern "Assignment.ipynb'" matched no files
    

                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --writer=<DottedObjectName>
        Writer class used to write the 
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files.
                    can only be used when converting one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current 
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy 
                of reveal.js. 
                For speaker notes to work, this must be a relative path to a local 
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and 
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of 
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    
    


```python

```
