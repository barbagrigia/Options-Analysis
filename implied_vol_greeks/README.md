
# Implied Volatility and Greeks

Importing required libraries


```python
import math as m
from scipy.stats import norm as n
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

%matplotlib inline
```


```python
df = pd.read_csv('SPX.csv')
print (df.shape)
df = df.drop_duplicates(subset=['Date','K'],keep='first')
df = df.reset_index(drop=True)
print (df.shape)
print (df.head(10))
```

    (721, 4)
    (600, 4)
            Date     T    K   Price
    0  3/20/2009  0.07  850    0.45
    1  3/20/2009  0.07  875    0.60
    2  3/20/2009  0.07  900    0.15
    3  3/20/2009  0.07  200  569.80
    4  3/20/2009  0.07  300  469.95
    5  3/20/2009  0.07  325  444.95
    6  3/20/2009  0.07  350  420.05
    7  3/20/2009  0.07  375  395.15
    8  3/20/2009  0.07  400  370.25
    9  3/20/2009  0.07  425  345.35


Considering only the call options for analysis.


```python
S = 770.05      #Stock Price of Google
r = 0.05      #Risk Free Rate
t = df['T']
Price = df['Price']
K = df['K']
```

Useful functions

#### Black Scholes Merton Model


```python
#Function to calculate the Option price using BSM
def BSMOption(S,K,t,r,sigma,type):
    
    d1 = (np.log(S/K)+(r+((sigma**2)/2))*t)/(sigma*np.sqrt(t))
    d2=d1-sigma*np.sqrt(t)
    
    if (type=='c'):    
        C = S*n.cdf(d1)-(K*np.exp(-r*t)*n.cdf(d2))
        return C
    else:    
        P = K*np.exp(-r*t)*n.cdf(-d2)-S*n.cdf(-d1)
        return P
```

### Root Finding Methods

#### Bisection Method


```python
#Calculating the implied Volatility using the Bisection Method
def bisect(S,K,r,t,types,MP):
    time1 = datetime.now()
    a = 0.0001       #Minimum Value
    b = 1       #Maximum Value
    N = 1       #Number of iterations
    tol = 10**-4
    
    f = lambda s:BSMOption(S,K,t,r,s,types)-MP         
    
    while N<=100:
        sig = (a+b)/2.0
        if f(sig)==0 or (b-a)/2<tol:
            time2 = datetime.now()
            t = time2-time1
            return sig
        N = N + 1
        if np.sign(f(sig))==np.sign(f(a)):
            a = sig
        else:
            b = sig
    print ("Did not converge")
    
```

#### Secant Method

This method is slightly faster than Newton method since there is no need to calculate the vega of the derivative which might sometimes be impossible to calculate. This is where we could use a Numerical approximation in Finite difference method.


```python
#Calculating the implied Volatility using the Secant Method
def secant(S,K,r,t,types,MP):
    time1 = datetime.now()
    x0 = 0.1
    xx = 1
    tolerance = 10**-7
    epsilon = 10**(-14)
    
    maxIterations = 100
    SolutionFound = False
    
    #Anonymous function to calculate the Implied volatility using the Secant Method
    f = lambda s:BSMOption(S,K,t,r,s,types)-MP         
    
    for i in range(maxIterations):
        y = f(x0)
        yprime = (f(x0)-f(xx))/(x0-xx)      
        
        if (abs(yprime)<epsilon):
            break
        
        x1 = x0 - y/yprime
        
        if (abs(x1-x0)<=tolerance*abs(x1)):
            SolutionFound = True
            break
        
        x0 = x1
    
    if (SolutionFound):
        time2 = datetime.now()
        t = time2 - time1
        return x1
    else:
        pass
```

#### Newton Method

It is faster than the Bisection method but to converge quickly, there is a need to make an approximate guess.


```python
#Calculating the implied Volatility using the Newton Method
def newton(S,K,r,t,types,MP):
    time1 = datetime.now()
    x0 = 1
    maxIterations =100
    
    epsilon = 10**-14
    tolerance = 10**-7
    solutionFound = False
    #Anonymous function to calculate the Implied volatility using the Newton Method
    f = lambda s:BSMOption(S,K,t,r,s,types)-MP  
    
    fprime = lambda sig:S*np.sqrt(t)*(1/np.sqrt(2*np.pi))*np.exp((-((np.log(S/K)+(r+((sig**2)/2))*t)/(sig*np.sqrt(t)))**2)/2)
    
    for i in range(maxIterations):
        
        y = f(x0)
        vega = fprime(x0)
        
        if (abs(vega)<epsilon):
            break
        
        x1 = x0 - y/vega
        if (abs(x1-x0)<=tolerance*abs(x1)):
            solutionFound = True
            break
        x0=x1
    
    if (solutionFound):
        time2 = datetime.now()
        t = time2 - time1
        return x1
    else:
        pass 
        
```

Calculating the implied vols using all the 3 above methods


```python
volb = [];vols=[];voln=[];bsm =[];
#Loop over options in the CSV file
for i in range(0,len(K)):
    volb.append(bisect(S,K.iloc[i],r,t.iloc[i],'c',Price.iloc[i]))
    bsm.append(BSMOption(S,K.iloc[i],t.iloc[i],r,volb[i],'c'))
    vols.append(secant(S,K.iloc[i],r,t.iloc[i],'c',Price.iloc[i]))
    voln.append(newton(S,K.iloc[i],r,t.iloc[i],'c',Price.iloc[i]))
```


```python
#Showcasing the values
df1 = pd.DataFrame({'Date':t,'K':K,'Bisection':volb,'Secant':vols,'Newton':voln,'MP':Price,'BSM':bsm})
df1.to_csv('vols_using_roots.csv',index=False)
```


```python
df1.head(15)
```




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
      <th>Date</th>
      <th>K</th>
      <th>Bisection</th>
      <th>Secant</th>
      <th>Newton</th>
      <th>MP</th>
      <th>BSM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.07</td>
      <td>850</td>
      <td>0.189717</td>
      <td>0.189712</td>
      <td>0.189712</td>
      <td>0.45</td>
      <td>0.450073</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.07</td>
      <td>875</td>
      <td>0.247695</td>
      <td>0.247752</td>
      <td>0.247752</td>
      <td>0.60</td>
      <td>0.599176</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.07</td>
      <td>900</td>
      <td>0.241348</td>
      <td>NaN</td>
      <td>0.241362</td>
      <td>0.15</td>
      <td>0.149927</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.07</td>
      <td>200</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>569.80</td>
      <td>570.748779</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.07</td>
      <td>300</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>469.95</td>
      <td>471.103585</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.07</td>
      <td>325</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>444.95</td>
      <td>446.203933</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.07</td>
      <td>350</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>420.05</td>
      <td>421.326020</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.07</td>
      <td>375</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>395.15</td>
      <td>396.494217</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.07</td>
      <td>400</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>370.25</td>
      <td>371.749258</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.07</td>
      <td>425</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>345.35</td>
      <td>347.152296</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.07</td>
      <td>440</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>332.499037</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.07</td>
      <td>450</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>320.55</td>
      <td>322.787035</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.07</td>
      <td>475</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>295.75</td>
      <td>298.759151</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.07</td>
      <td>490</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>280.95</td>
      <td>284.555405</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.07</td>
      <td>500</td>
      <td>0.999939</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>271.10</td>
      <td>275.192793</td>
    </tr>
  </tbody>
</table>
</div>



The NaNs are due to the Market Price being smaller than the Theoretical Price (Black Scholes price).

### Volatility Smile


```python
vols = pd.read_csv('vols_using_roots.csv',index_col=['Date'])
#Dropping nans
vols.dropna(inplace=True)
vols.head()
```




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
      <th>K</th>
      <th>Bisection</th>
      <th>Secant</th>
      <th>Newton</th>
      <th>MP</th>
      <th>BSM</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.07</th>
      <td>850</td>
      <td>0.189717</td>
      <td>0.189712</td>
      <td>0.189712</td>
      <td>0.45</td>
      <td>0.450073</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>875</td>
      <td>0.247695</td>
      <td>0.247752</td>
      <td>0.247752</td>
      <td>0.60</td>
      <td>0.599176</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>550</td>
      <td>0.514636</td>
      <td>0.514623</td>
      <td>0.514623</td>
      <td>222.15</td>
      <td>222.150038</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>560</td>
      <td>0.549178</td>
      <td>0.549174</td>
      <td>0.549174</td>
      <td>212.45</td>
      <td>212.450028</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>575</td>
      <td>0.564680</td>
      <td>0.564731</td>
      <td>0.564731</td>
      <td>197.95</td>
      <td>197.949489</td>
    </tr>
  </tbody>
</table>
</div>




```python
vols = vols.sort_values(by=['Date','K','Newton'],ascending=[True,True,True])
vols.head()
```




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
      <th>K</th>
      <th>Bisection</th>
      <th>Secant</th>
      <th>Newton</th>
      <th>MP</th>
      <th>BSM</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.07</th>
      <td>550</td>
      <td>0.514636</td>
      <td>0.514623</td>
      <td>0.514623</td>
      <td>222.15</td>
      <td>222.150038</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>560</td>
      <td>0.549178</td>
      <td>0.549174</td>
      <td>0.549174</td>
      <td>212.45</td>
      <td>212.450028</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>575</td>
      <td>0.564680</td>
      <td>0.564731</td>
      <td>0.564731</td>
      <td>197.95</td>
      <td>197.949489</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>580</td>
      <td>0.567487</td>
      <td>0.567483</td>
      <td>0.567483</td>
      <td>193.15</td>
      <td>193.150043</td>
    </tr>
    <tr>
      <th>0.07</th>
      <td>590</td>
      <td>0.573712</td>
      <td>0.573712</td>
      <td>0.573712</td>
      <td>183.65</td>
      <td>183.650002</td>
    </tr>
  </tbody>
</table>
</div>




```python
vols = vols[(vols['K']>600) & (vols['K']<1200)]
```

#### 2- D Plot


```python
#Aggregate
plt.plot(vols.loc[0.15]['K'],vols.loc[0.15]['Newton'],'g')
plt.plot(vols.loc[0.22]['K'],vols.loc[0.22]['Newton'],'r')
plt.plot(vols.loc[0.57]['K'],vols.loc[0.57]['Newton'],'b')
plt.xlabel("K")
plt.ylabel("Newton. Implied Volatility")
plt.title("Volatility Smile")
plt.legend([0.15,0.22,0.57])

plt.savefig('Volatility Smile.png')
```


![png](output_28_0.png)


#### 3- D Plot


```python
vols = vols.loc[[0.15,0.22,0.57]]
vols.index.unique()
```




    Float64Index([0.15, 0.22, 0.57], dtype='float64', name='Date')




```python
fig = plt.figure(figsize=(9,6))
ax = fig.gca(projection='3d')
d3 = ax.scatter(vols['K'],vols.index,vols['Newton'],s=20,c=None,depthshade=True)

ax.set_xlabel('Strike')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Implied Vol')
```




    Text(0.5, 0, 'Implied Vol')




![png](output_31_1.png)


### Greeks

Greeks are measures which can tell the sensitivity of an option's price based on different factors.


$
Delta:\Delta = \partial C / \partial S
\\ Gamma: \gamma = \partial^2 C / \partial S^2
\\ Vega: \nu = \partial C / \partial \sigma
$

#### Analytical Solution


```python
#Greeks using the Analytical Formula       
def Greeks(S,K,t,r,sigma):
    d1 = (m.log(S/K)+(r+(sigma**2/2))*t)/(sigma*m.sqrt(t))
    NDashd1 = (1/m.sqrt(2*m.pi))*m.exp((-d1**2)/2) 
    Delta = n.cdf(d1)
    Gamma = NDashd1/(S*sigma*m.sqrt(t))
    Vega = S*m.sqrt(t)*NDashd1
    return Delta,Gamma,Vega
```


```python
greeks = []
for i in range(vols.shape[0]):
    greeks.append(Greeks(S,vols['K'].iloc[i],vols.index[i],r,vols['Newton'].iloc[i]))
```


```python
analytical_df = pd.DataFrame(greeks)
analytical_df.columns = ['Delta','Gamma','Vega']
analytical_df.head()
```




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
      <th>Delta</th>
      <th>Gamma</th>
      <th>Vega</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.954890</td>
      <td>0.000816</td>
      <td>19.349468</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.949995</td>
      <td>0.000889</td>
      <td>21.014139</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.945056</td>
      <td>0.000963</td>
      <td>22.645727</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.939441</td>
      <td>0.001044</td>
      <td>24.446562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.933769</td>
      <td>0.001126</td>
      <td>26.210948</td>
    </tr>
  </tbody>
</table>
</div>



#### Finite difference approximations

Although there is no need of the approximations method since we have a closed form solutions for the Greeks. It is slightly faster and sometimes speed might be more important than the exact solution.


```python
#Greeks using the FDM Approximations
def GreeksFDM(S,K,t,r,sigma):
    
    #price and sigma increment
    d_s = 0.001
    
    CDelta = (BSMOption(S+d_s,K,t,r,sigma,'c') -BSMOption(S,K,t,r,sigma,'c'))/d_s
    
    CGamma = (BSMOption(S+d_s,K,t,r,sigma,'c') -2*BSMOption(S,K,t,r,sigma,'c')+BSMOption(S-d_s,K,t,r,sigma,'c'))/(d_s**2)
    
    CVega = (BSMOption(S,K,t,r,sigma+d_s,'c') -BSMOption(S,K,t,r,sigma,'c'))/d_s
    
    return CDelta,CGamma,CVega
```


```python
greeks = []
for i in range(vols.shape[0]):
    greeks.append(GreeksFDM(S,vols['K'].iloc[i],vols.index[i],r,vols['Newton'].iloc[i]))
```


```python
fdm_df = pd.DataFrame(greeks)
fdm_df.columns = ['Delta','Gamma','Vega']
fdm_df.head()
```




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
      <th>Delta</th>
      <th>Gamma</th>
      <th>Vega</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.954890</td>
      <td>0.000816</td>
      <td>19.393717</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.949995</td>
      <td>0.000889</td>
      <td>21.059462</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.945056</td>
      <td>0.000963</td>
      <td>22.692015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.939441</td>
      <td>0.001044</td>
      <td>24.493584</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.933770</td>
      <td>0.001126</td>
      <td>26.258601</td>
    </tr>
  </tbody>
</table>
</div>


