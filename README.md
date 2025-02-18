# B-VAR with TVP for Macro-Economic Data

This is a personal project. From the following academic review [1], I became inspired to explore a new method of Bayesian modelling, which was to model a vector time-series as an autoregressive process with uncertainties. This method sees plenty of use in macroeconomics[1-6], as data is quite sparse due to the large timescale underlying the economy (on the order of months, quarters, years), and many of the macro-economic indicators have complex relations. The challenge then becomes how one deals with the data sparsity, the time-varying economic volatility and the notoriously slow scaling of BVAR models wrt the number of modelled indicators. Luckily, years of literature has yielded some clever solutions to these challenges[2][3][4], and I try my hand at implementing a common benchmark model within the field, which is the B-VAR with a simple stochastic volatility model and the so-called Minnesota prior[5][6]. I also wanted to model all of this within stan, using python and the cmdstanpy/arviz packages.

## Data

I used a dataset consisting of many US-centric macro-economic indicators on a month-by-month basis, via the Federal Reserve Bank of St. Louis, which was graciously compiled by M. W. McCracken [7]. For data preprocessing in particular, I drew inspiration from [3] (Table 5), where I performed identical data transformations to evoke quasi stationarity and also used standard scaling to center and normalize the features. 

## Model

Let $\mathbf{y}_t$ b e a $D$-dimensional real vector of the transformed macro-economic indicators at time index $t=1,2,...,T$. We propose the following autoregressive model for $\mathbf{y}_t$  -- VAR(P).
```math
\mathbf{y}_t = \mathbb{A}_1 \cdot \mathbf{y}_{t-1} + \mathbb{A}_2 \cdot \mathbf{y}_{t-2} + \cdots + \mathbb{A}_P \cdot \mathbf{y}_{t-P} + \mathbf{\epsilon}_t
```
Where $\mathbb{A}_p \in \mathbb{R}^{D\times D}$ is a matrix of parameters to be fit, and $\mathbf{\epsilon}_t$ represents a noise term. We can already see the afformentioned problems rearing their head. First of all, this is already $D^2 P$ parameters (in a reality where an economy has D = O(100) and we are limited to data N = O(1000) or less.). This is all without modelling the noise term, which is not stationary and the noise elements are decidedly NOT independent. Here is where we get to see some interesting modelling decisions. First, let's apply the Minnesota prior to $\mathbb{A}_p$, thereby reducing the effective number of parameters (but importantly, this is not speeding up sampling -- to do so, I'd recommend reading [2-4] for 3 different methods -- I will not be using these advanced sampling/modelling techniques). 

## Minnesota Prior
We propose the following prior on $\mathbb{A}_ p $. First, define notation $A_ {pij} := (\mathbb{A}_ p)_{ij}$. We set the prior means of the elements ($\mu _{pij}$) as follows.
```math
\mu_{pij} = \begin{cases} 1 : \text{if p=1 and i=j},\\ 0 : \text{otherwise}\end{cases}
```
What this does is effectively set the prior belief that all of the economic indicators are independent simple random walks. This is because the diagonals are 0, and all elements are 0 which are not the diagnoal of the preceding timestep. Think of $\mathbb{A}_1 \ \text{Ident}(D)$ and $\mathbb{A} _ {p \neq 1} = \mathbb{0}_D$ as a the prior. When the model is being fit, it is deviating slightly from this prior, mitigating overfitting. 

We also set the variances of the priors in a clever way. We introduce a regularization parameter $\lambda > 0$ which modulates regularization, and we scale the variance of higher order $\mathbb{A}_ p$ to promote our prior belief that recency is more predictive. We set the variances $V_ {pij}$ as follows:
```math
V_{pij} = \frac{\lambda ^ 2}{p^2} \frac{\sigma_i^2}{\sigma_j^2}
```
Although we have scaled our features before the fit so that $\sigma_i \approx \sigma_j \approx 0$ so we set them to 1 in the equation.
## Stochastic Volatility
We also must model the volatility of the economy, encapsulated in the term $\mathbf{\epsilon}_t$. We employ the stochastic volatility model as outlined by [7] and the notation from [3]. It essentially comes down to proposing some ansatz on $\epsilon$ which has been motivated by years of data and analysis of the US economy. We propose the following prior on $\epsilon$:
```math
\mathbf{\epsilon}_t \sim \text{MvNormal} (\mathbf{0}_D, \Sigma_t) \text{ where } \Sigma_t = \exp(h_t) \Omega
```
Where $\Omega \in \mathbb{R}^{D \times D}$ represents the static covariances between the noise of the assets and the prefactor $\exp(h_t)$ respresents the changing economic volatility. In particular, $h_t$ is modelled as a random walk with regularization. 
```math
h_t \sim \text{Normal} (\phi h_{t-1}, \sigma_h)
```
Where $\sigma_h > 0, \phi \in [0,1]$ are to be fitted. With this framework, we are able to implement the var completely in stan.

## Interesting Plots
![image](https://github.com/user-attachments/assets/7a535cb2-7c41-4129-b483-cd8e0647c50d)
The above figure depicts the posterior predictive density of $h_t$ from 2002 to 2025. What we see is a pretty clear indicator of 2008 housing market crash and covid-19's effect on the economy. I also plotted the out of sample simulation of the next 12 months of $h$. Of course, being a random walk, there is little to no structure, and we just see an ambient diffusion as it approaches a steady state. Still interesting though.
![image](https://github.com/user-attachments/assets/2c4e2fb8-b7e8-45d4-bf36-eb7fab8aa7d2)

arviz posterior trace plots of the stochastic volatility parameters.
![image](https://github.com/user-attachments/assets/85fe4407-b6e6-43f2-a8d4-5b572aaf7e51)

The static covariance matrix $\Omega$ visualization of 20 economic indicators. Notice that some values are >1, meaning the standard scaling procedure was imperfect. This is because it was scaled before I trimmed the data to be greater than 2002.

## Bibliography
[1] G. M. Martin, D. T. Frazier, W. Maneesoonthorn, R. Loaiza-Maya, F. Huber, G. Koop, J. Maheu, D. Nibbering, and A. Panagiotelis, "Bayesian forecasting in economics and finance: A modern review," Int. J. Forecast., vol. 40, no. 2, pp. 811–839, 2024. [Online]. Available: https://doi.org/10.1016/j.ijforecast.2023.05.002.

[2] F. Huber and M. Pfarrhofer, "Dynamic shrinkage in time‐varying parameter stochastic volatility in mean models," J. Appl. Econom., vol. 36, no. 2, pp. 262–270, Mar. 2021. [Online]. Available: https://ideas.repec.org/a/wly/japmet/v36y2021i2p262-270.html.

[3] J. Chan and Y. Qi, "Large Bayesian tensor VARs with stochastic volatility," arXiv preprint, arXiv:2409.16132, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2409.16132.

[4] N. Hauzenberger, F. Huber, G. Koop, and L. Onorante, "Fast and flexible Bayesian inference in time-varying parameter regression models," J. Bus. Econ. Stat., vol. 40, no. 4, pp. 1904–1918, 2021. [Online]. Available: https://doi.org/10.1080/07350015.2021.1990772.

[5] R. Litterman, "Techniques of forecasting using vector autoregressions," Federal Reserve Bank of Minneapolis Working Paper, no. 115, 1979.

[6] R. Litterman, "Specifying VAR's for macroeconomic forecasting," Federal Reserve Bank of Minneapolis Staff Report, no. 92, 1984.

[7] A. Carriero, T. E. Clark, and M. G. Marcellino, "Common drifting volatility in large Bayesian VARs," J. Bus. Econ. Stat., vol. 34, no. 3, pp. 375–390, 2016.
    M. W. McCracken, "FRED-MD," FRED, Federal Reserve Bank of St. Louis, Jan. 2024. Accessed: Feb. 17, 2025. [Online]. Available: [https://fred.stlouisfed.org/series/CPIAUCSL](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
