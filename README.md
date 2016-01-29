Utils for fast work with Pandas:

* Multi core pd.apply function

Example:



```python
%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
df = pd.DataFrame({'a': np.random.randn(1000000),
     'b': np.random.randn(1000000),
     'N': np.random.randint(100, 1000000, (1000000)),
     'x': 'x'})

df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>a</th>
      <th>b</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>74566</td>
      <td>0.312583</td>
      <td>-0.180284</td>
      <td>x</td>
    </tr>
    <tr>
      <th>1</th>
      <td>431825</td>
      <td>-0.701312</td>
      <td>0.822490</td>
      <td>x</td>
    </tr>
    <tr>
      <th>2</th>
      <td>258068</td>
      <td>0.120284</td>
      <td>0.031232</td>
      <td>x</td>
    </tr>
    <tr>
      <th>3</th>
      <td>465223</td>
      <td>0.623214</td>
      <td>-0.019660</td>
      <td>x</td>
    </tr>
    <tr>
      <th>4</th>
      <td>116829</td>
      <td>-1.569378</td>
      <td>0.380467</td>
      <td>x</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>500073.402014</td>
      <td>0.000791</td>
      <td>-0.000226</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288727.641253</td>
      <td>1.000553</td>
      <td>1.001541</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100.000000</td>
      <td>-5.180971</td>
      <td>-4.845733</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>250056.750000</td>
      <td>-0.673639</td>
      <td>-0.675882</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>500067.500000</td>
      <td>-0.000393</td>
      <td>-0.001142</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>750587.250000</td>
      <td>0.675965</td>
      <td>0.675489</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999998.000000</td>
      <td>4.571504</td>
      <td>5.068164</td>
    </tr>
  </tbody>
</table>
</div>




```python
%time df['test1'] = df.apply(lambda x: (x['a']*10342)/23, axis=1)
```

    CPU times: user 23.8 s, sys: 210 ms, total: 24.1 s
    Wall time: 24 s



```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>a</th>
      <th>b</th>
      <th>test1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>500073.402014</td>
      <td>0.000791</td>
      <td>-0.000226</td>
      <td>0.355555</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288727.641253</td>
      <td>1.000553</td>
      <td>1.001541</td>
      <td>449.900646</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100.000000</td>
      <td>-5.180971</td>
      <td>-4.845733</td>
      <td>-2329.634669</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>250056.750000</td>
      <td>-0.673639</td>
      <td>-0.675882</td>
      <td>-302.903441</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>500067.500000</td>
      <td>-0.000393</td>
      <td>-0.001142</td>
      <td>-0.176596</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>750587.250000</td>
      <td>0.675965</td>
      <td>0.675489</td>
      <td>303.948933</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999998.000000</td>
      <td>4.571504</td>
      <td>5.068164</td>
      <td>2055.586513</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Work with multi core
from stutils import pandas
```


```python
#Create instanse
worker = pandas.PdFastApply()
```


```python
#We can use 16 process.
worker.rep()
```

    Number max process: 16
    Number splits: 0
    Number current process: 0



```python
#Split our dataframe
worker.df_split(df)
```


```python
#We can see current number of splits
worker.rep()
```

    Number max process: 16
    Number splits: 16
    Number current process: 0



```python
#Apply our function in no wait mode
worker.df_apply(lambda x: x['a']*10342/23, 'test2')
```


```python
worker.ns.df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>a</th>
      <th>b</th>
      <th>test2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>499891.865848</td>
      <td>-0.000106</td>
      <td>0.000781</td>
      <td>-0.047719</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288507.598777</td>
      <td>0.998211</td>
      <td>1.000105</td>
      <td>448.847715</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100.000000</td>
      <td>-4.685454</td>
      <td>-4.922438</td>
      <td>-2106.824567</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>250067.500000</td>
      <td>-0.672058</td>
      <td>-0.672937</td>
      <td>-302.192225</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>500201.500000</td>
      <td>-0.000203</td>
      <td>0.000658</td>
      <td>-0.091090</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>749485.000000</td>
      <td>0.672126</td>
      <td>0.676078</td>
      <td>302.222898</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999999.000000</td>
      <td>4.750408</td>
      <td>5.074973</td>
      <td>2136.031213</td>
    </tr>
  </tbody>
</table>
</div>
