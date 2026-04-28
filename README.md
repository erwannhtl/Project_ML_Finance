# Project_ML_Finance

\documentclass[12pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french,english]{babel}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{float}
\usepackage{appendix}


\begin{document}

\begin{titlepage}
    \centering
    \vspace*{1cm}
    
    \begin{flushleft}
        \includegraphics[width=0.25\textwidth]{logo.png} 
    \end{flushleft}   
    
    \centering
    \vspace*{1.5cm}
    
    \rule{\linewidth}{0.5mm} \\[0.4cm]
    
    % Main Title
    {\huge \bfseries Deep learning volatility \par}
    \vspace{0.3cm}
    
    % Subtitle/Paper Review Reference
    {\large \textit{A review of the paper by Bayer, Horvath, Muguruza, Stemper, and Tomas} \par}
    \rule{\linewidth}{0.5mm} \\[1.5cm]
    
    \vspace{2cm}

    % Authors Section
    {\large Erwann \textsc{Hotellier} \\ Capucine \textsc{Rousset} \\ Taddéo \textsc{Vivet} \par}

    \vfill

    % Bottom Date
    {\large April 2026 \par}
    \vspace*{1cm}
    
\end{titlepage}

% This command ensures the report starts on the next page (page 1)
\newpage
% =================================================
\section{Introduction}

In financial mathematics, one of the most important tasks is model calibration: finding the parameters of a mathematical model so that the prices it produces match the prices observed in real markets. For simple models, such as the famous Black-Scholes model, this task is fast because an exact pricing formula exists. However, more realistic models, such as the family of \textit{rough volatility models}, require slow numerical simulation methods to compute prices, which makes calibration far too slow for practical use in the financial industry.
 
This report presents and discusses the paper \textit{``Deep Learning Volatility''} by Horvath, Muguruza, and Tomas (2019). The authors propose a framework that uses deep neural networks to solve the calibration bottleneck described above. Their key idea is to train a neural network offline to act as a fast pricing calculator, and then use this learned approximation to perform calibration very efficiently. As we will explain, this two-step approach combines the theoretical accuracy of rough volatility models with the computational speed required for real-world applications.

\section{Literature Review}

\subsection{Historical Context and Prior Work}

Over the years, researchers have slowly improved on the simple Black and Scholes model in order to better explain prices moves. More recently, they have demonstrated that market volatility is not constant, but rather stochastic and "rough". Bayer, Friz, and Gatheral (2015) took an important step in that direction by introducing the rough Bergomi model, where the variance process $V_t$ is driven by a fractional Brownian motion. The dynamics of this model are described using a stochastic exponential $\mathcal{E}$ and a Hurst parameter $H < 1/2$, which controls the roughness of the paths:
$$ V_t = \xi_0(t)\mathcal{E}\left(\nu\sqrt{2H}\int_0^t(t-s)^{H-1/2}dZ_s\right) $$
where $\xi_0(t)$ is the forward variance curve (the market's expectation of future variance) and $\nu$ controls the volatility of volatility. 

This model accurately captures important features of real market data, such as the \textit{volatility skew}. However, it has one main limitation: it does not admit a closed-form pricing formula. Computing an option price therefore requires Monte Carlo simulation, which is computationally expensive. This creates what the authors call a \textit{calibration bottleneck}, making the rough Bergomi model too slow for real-time use in financial institutions.

In order to solve those complex pricing equations, researchers have long looked toward machine learning as a way to speed up the process. Hutchinson, Lo, and Poggio (1994) proposed using artificial neural networks as non-parametric tools for option pricing. Instead of relying on a specific mathematical model (like Black-Scholes) to define market dynamics, they trained a neural network $f_{NN}$ to predict the call option price $C$ directly from observable market variables. Specifically, using the principle that option prices depend on the ratio of the current asset price $S$ to the strike price $K$ (known as \textit{moneyness}) and the time to maturity $T$, they approximated:
\[
\frac{C}{K} = f_{\mathrm{NN}}\!\left(\frac{S}{K},\, T;\, w\right),
\]
where $w$ denotes the learned weights of the network and $C$ is the call option price. While this data-driven approach was innovative, it had an important limitation: by bypassing any stochastic model entirely, the method lost its connection to financial theory, making it difficult for risk managers to interpret the results or to ensure that the prices produced by the network were free of arbitrage opportunities.

To bring machine learning into modern stochastic modeling, Hernandez (2017) proposed using neural networks to solve the calibration bottleneck. He noted that traditional calibration consists of finding the parameters $\theta$ (in a set $S \subseteq \mathbb{R}^n$) that minimize a cost function between a set of $N$ market quotes $\{Q^{mkt}\}$ and the model's pricing function $f(\theta)$:
$$ \hat{\theta} = \arg \min_{\theta \in S \subseteq \mathbb{R}^n} \text{Cost}\left( \{Q^{mkt}\}; f(\theta) \right) $$
This can be a computationally intensive process, and Hernandez proposed viewing the calibration problem as a direct mathematical function, which reduces calibration time. He trained a neural network $W$ to map the $N$ market quotes directly to the $n$ model parameters:
$$ W : \mathbb{R}^N \rightarrow S \subseteq \mathbb{R}^n $$
However, directly finding parameters from market prices is an "inverse problem." Inverse problems are often mathematically unstable. Because multiple different sets of parameters might produce very similar prices, a direct neural network approach can struggle to find a stable, unique solution.

\subsection{Contribution of the Paper}

The papers described above left an important gap: the industry needed the theoretical accuracy of rough volatility models, but without the slow speed of Monte Carlo simulations or the mathematical instability of direct neural network inversion. 

The paper by Horvath, Muguruza, and Tomas closes this gap by introducing a two-step "Deep Calibration" framework. Instead of asking the neural network to directly invert the market data to find parameters, they train the network off-line to learn the forward pricing map. This means the network simply generates prices from inputed model parameters in milliseconds. Then, the actual calibration is solved on-line as a deterministic optimization problem. This separation perfectly avoids the instability of direct inversion while dramatically reducing the calibration time of complex models.

Furthermore, the authors solve the issue of parameter uniqueness by introducing a grid-based "image" approach. Instead of predicting a single option price at a time, the neural network is trained to output a complete grid of implied volatilities (they use a grid of 8 maturities and 11 strikes) all at once. By treating the entire volatility surface as a collection of pixels like in an image, the network learns the structural relationships between consecutive points. This global view ensures that the mapping from parameters to the volatility surface is accurate and reliable.


\section{Technical Summary}

\subsection{The Deep Calibration Mathematical Framework}

In order to implement the two-step framework detailed above, this section details the mathematical implementation used to construct the forward pricing map. The main objective here is to train a neural network, $\tilde{F}$, that maps a vector of model parameters $\Theta$ directly to a complete implied volatility surface, approximating the true numerical pricing map $\tilde{P}$:
\begin{equation}
    \tilde{F}(\Theta, \zeta) \approx \tilde{P}(\mathcal{M}(\Theta, \zeta))
\end{equation}
Rather than calculating a single option price, the authors output a structured $n \times m$ grid of implied volatilities. The neural network predicts the entire surface simultaneously, which allows to better capture the evolution of the volatility across options with different characteristics:
\begin{equation}
    F^*(\theta) = \{\sigma_{BS}^{\mathcal{M}(\theta)}(T_i, k_j)\}_{i=1, j=1}^{n, m}
\end{equation}
In their numerical experiments, the authors use a grid of $n=8$ maturities and $m=11$ strikes. This yields a matrix of 88 output points per forward evaluation. Because this structure forces the network to evaluate the entire surface at once, it severely restricts the possibility of multiple model parameter combinations yielding the exact same output grid, thereby guaranteeing a unique and stable solution during the calibration phase.

\subsection{The Parameter Space of the Volatility Models}

To ensure their framework is universally applicable, the authors test it across three distinct stochastic volatility models. While the continuous-time dynamics of the rough Bergomi model were established in the literature review, implementing it practically requires discretizing the parameter space. The authors treat the forward variance curve, $\xi_0(t)$, not as a single constant, but as a piecewise constant function (a staircase) with 8 distinct time segments. Therefore, the neural network must simultaneously calibrate a total of 11 specific parameters ($p=11$) to successfully fit the rough Bergomi model to the market.

The second model tested by the authors is the standard 1-Factor Bergomi model. This model represents a more traditional, smooth volatility surface. Its parameter vector is defined as $\theta = (\xi_0, \beta, \eta, \rho)$, and the variance process is expressed as:
\begin{equation}
    V_t = \xi_0(t)\mathcal{E}\left(\eta\int_0^t \exp(-\beta(t-s))dZ_s\right)
\end{equation}
Similar to the rough version, the network must find the piecewise forward variance curve $\xi_0$ and the correlation $\rho$. However, instead of a roughness parameter, it searches for $\eta$, representing the volatility of volatility, and $\beta$, the speed of mean reversion that dictates how quickly unexpected market spikes return to historical averages.

Furthermore, for their final experiments on artificial intelligence model recognition, the authors incorporate the classic Heston model. The parameter vector for Heston is defined as $\theta = (a, b, v, \rho)$, with the variance process described by the following differential equation:
\begin{equation}
    dV_t = a(b - V_t)dt + v\sqrt{V_t}dZ_t
\end{equation}
When calibrating the Heston model, the neural network searches for $a$, the speed of mean reversion; $b$, the long-term baseline variance; and $v$, the volatility of volatility. 

\subsection{Network Architecture and Offline Training}

To execute this high-dimensional approximation efficiently, the authors designed a fully connected feed-forward neural network. The architecture begins with an input layer for the model parameters, followed by four hidden layers containing exactly 30 nodes each, and finally an output layer of 88 nodes corresponding to the volatility grid. The choice of 4 layers is motivated by mathematical theory, which dictates that shallower networks often struggle to efficiently approximate highly complex functions. This design allows the entire network to learn only 5,878 internal weights and biases, ensuring fast execution times and preventing the model from overfitting the training data.

A critical technical choice in this Neural Network is the use of the Exponential Linear Unit (ELU) activation function for the hidden layers:
\begin{equation}
  \sigma_{Elu}(x) =
  \begin{cases}
    x & \text{if } x \geq 0, \\
    \alpha(e^x - 1) & \text{if } x < 0.
  \end{cases}
\end{equation}
Unlike the standard ReLU function, which has a sharp, non-differentiable corner at zero, the ELU function is perfectly smooth. This mathematical smoothness is crucial because it allows quantitative analysts to calculate the exact mathematical slopes, or derivatives, of the network output with respect to the input parameters. This property is essential for the gradient-based optimisers used in the calibration step described below.

The network is trained entirely offline using a massive synthetic dataset. The authors computationally generated 68,000 random parameter combinations for training and 12,000 for testing, using Monte Carlo simulations with 60,000 sample paths to calculate the true volatility grid for each. To further stabilise training and improve convergence, the authors apply three regularisation techniques: \textit{early stopping}, which halts training if performance on the test set does not improve for 25 consecutive epochs; \textit{parameter normalisation}, which rescales all model parameters to the interval $[-1, 1]$; and \textit{implied volatility normalisation}, which standardises the output by subtracting the empirical mean and dividing by the standard deviation. The network learns by minimizing the Mean Squared Error (MSE) between its own predictions, $F(\theta_u, w)_{ij}$, and the true Monte Carlo benchmark, $F^*(\theta_u)_{ij}$, across the entire $8 \times 11$ grid:
\begin{equation}
    \hat{w} = \arg\min_{w \in \mathbb{R}^n} \sum_{u=1}^{N_{Train}} \sum_{i=1}^n \sum_{j=1}^m (F(\theta_u, w)_{ij} - F^*(\theta_u)_{ij})^2
\end{equation}

\subsection{Online Calibration and Optimization}

Once the neural network is fully trained and its weights $\hat{w}$ are set, the online calibration step performs a deterministic search for the model parameters $\hat{\theta}$. The optimizer aims to minimize the squared distance between the network's predicted grid and the implied volatilities actually observed in the real-world market, $\sigma_{BS}^{MKT}$:
\begin{equation}
    \hat{\theta} = \arg\min_{\theta \in \Theta} \sum_{i=1}^n \sum_{j=1}^m (\tilde{F}(\theta)_{ij} - \sigma_{BS}^{MKT}(T_i, k_j))^2
\end{equation}
To solve this equation rapidly, the authors leverage the exact gradients provided by the ELU activation function and employ gradient-based optimisers, with the Levenberg-Marquardt algorithm identified as the most balanced choice in terms of speed and convergence. To assess robustness, the authors also systematically compare six other optimisers, including L-BFGS-B, SLSQP, BFGS, COBYLA, Differential Evolution, and Nelder-Mead. The key finding from this comparison is that gradient-based methods are dramatically faster but may converge to a local rather than a global minimum, whereas gradient-free methods such as Differential Evolution are slower but more reliable for finding the globally optimal solution. This trade-off means that practitioners must choose their optimiser based on the complexity of the parameter space at hand.

\subsection{Empirical Performance and Extensions}

The numerical experiments confirm that this deep learning framework successfully maintains the precision of traditional numerical methods like Monte Carlo, while eliminating the computational bottleneck. The average relative error between the neural network's prediction and the true Monte Carlo price remained consistently below 0.5 percent. Additionally, generating a full 88-point volatility surface took the neural network only 30.9 microseconds, compared to roughly 500,000 microseconds for the traditional Monte Carlo method. This results in an impressive computational speedup factor of 9,000 to 16,000 times. Furthermore, the network generalises well to parameter combinations it has never seen during training: the approximation accuracy measured on the out-of-sample test set remains within the same range as on the synthetic training set, confirming that the framework does not simply memorise the training data but genuinely learns the structure of the pricing map.

To prove the framework's robustness in industry practice, the authors performed a historical calibration using actual options data for the S\&P 500 index from 2010 to 2019. The neural network successfully and rapidly tracked the dynamics of the stock market day by day. The framework also proved highly adaptable to exotic derivatives. By simply replacing the grid of strike prices with a grid of barrier levels, the network accurately priced digital down-and-in and down-and-out barrier options with average absolute errors below 10 basis points.

Finally, the authors extended their grid-based approach to perform automatic model recognition. They trained a new classification network using a synthetic dataset of 320,000 examples to evaluate the overall shape of a market's volatility surface and predict which mathematical model created it. This network successfully identified the distinct models and accurately estimated the mixing weights when multiple models were combined, correctly expressing Bayesian uncertainty when surfaces overlapped. This experiment demonstrates that deep learning can be used not only to calibrate models at unprecedented speeds but also to automatically select the optimal mathematical model for any given market environment.



\section{Numerical Illustration}

In this section, we will present our implementation of this two-step calibration using the rBergomi with piecewise variance as our reference model. We tested our model on EUR/USD option data from  start date = 2021-06-08 to end date = 2026-04-13. We also introduced a new penalization function. The code used to generate our results is based off the GitHub provided by the authors (\textit{NN-StochVol-Calibrations}).

\subsection{Generating the Synthetic Dataframe}
The first step of this calibration was to produce the synthetic dataset. This is a crucial
part of our project, since the overall quality and fit of our Neural Network is benchmarked on the
quality of our initial dataset. For this, we generate $N = {60000}$ independent
parameter configurations sampled uniformly from the valid parameter space of the Rough Bergomi
model. Specifically, for each simulated market scenario, we draw the Hurst exponent
$H \sim \mathcal{U}(0.025,\, 0.5)$, the volatility of volatility
$\nu \sim \mathcal{U}(0.3,\, 2.5)$, the spot-variance correlation
$\rho \sim \mathcal{U}(-0.8,\, 0.8)$, and a piecewise-constant forward variance curve
comprising nine distinct levels $\xi_i \sim \mathcal{U}(0.001,\, 0.010)$ corresponding to
each option maturity $T_i$. These bounds were chosen to reflect realistic EUR/USD dynamics.

We updated the maturity grid to nine tenors specifically calibrated to the short-dated
structure of the FX options market we gathered from the CME dataset:
\[
    T \;\in\;
    \{\,0.019,\ 0.038,\ 0.057,\ 0.082,\ 0.164,\ 0.246,\ 0.493,\ 0.739,\ 1.000\,\}
    \text{ years,}
\]
corresponding approximately to 1W, 2W, 3W, 1M, 2M, 3M, 6M, 9M, and 1Y. 

For each randomised parameter set, we employ a Monte Carlo simulator utilising
10000 paths and a discretisation of 500 time steps per year to
price European call options across a fixed grid of 9 maturities and 11 strikes. The strike grid is not fixed in absolute terms but is instead constructed \emph{adaptively per maturity} via a standardised log-moneyness parameterisation:
\[
    K_j \;=\; S_0 \cdot \exp\!\left(z_j \cdot \sigma_{\text{ref}} \cdot \sqrt{T}\right),
    \qquad
    z_j \;\in\; \{-2.5,\,-2.0,\,\ldots,\,2.0,\,2.5\},
\]
where $\sigma_{\text{ref}} = 0.08$ is a fixed ATM reference volatility for EUR/USD. This construction is essential for numerical stability across the wide range of maturities considered.

To see why, consider what a fixed absolute strike grid would imply at a very short maturity such as $T = 0.019$ (roughly one week). A strike placed one standard deviation away from spot in a Black--Scholes sense corresponds to a log-moneyness of only
\[
    \sigma_{\text{ref}}\sqrt{T} \;\approx\; 0.08 \times \sqrt{0.019} \;\approx\; 0.011,
\]
that is, barely $1.1\%$ away from spot. By contrast, the adaptive grid keeps every strike at a financially meaningful distance from spot: at maturity $T$, the $j$-th strike always lies exactly $z_j$ standard deviations away under the reference log-normal measure, so that the moneyness axis automatically \emph{contracts} for short maturities and \emph{expands} for long ones. The corresponding log-moneyness used as the horizontal axis in all implied volatility plots is:
\[
    m_j(T) \;=\; z_j \cdot \sigma_{\text{ref}} \cdot \sqrt{T}.
\]
This ensures that, across all maturities, every point on the smile grid carries a comparable amount of pricing information and that the network is never asked to interpolate in regions devoid of meaningful market signal.

Furthermore, this adaptive parameterisation perfectly mirrors the quoting conventions of the Foreign Exchange (FX) options market, ensuring consistency between our numerical scheme and the empirical data. In our historical dataset, implied volatilities are gathered at standardised delta pillars (e.g., 10D, 25D, 50D) using mid-market quotes. A formal proof that delta-quoted data inherently forms such an adaptive grid is provided in Appendix~\ref{app:delta_grid}.

The resulting $9 \times 11$ implied volatility surface is subsequently flattened into a
99-dimensional vector. These 99 implied volatilities are then concatenated with the 12
generating parameters to form a single 111-dimensional row in our final synthetic
\texttt{DataFrame}, providing a direct mapping from Rough Bergomi parameters to the
observable implied volatility surface. Due to computational constraints, we were unable to
simulate the same volume of Monte Carlo paths and parameter combinations as the original
authors, which inevitably introduces a higher degree of simulation noise into our synthetic
training data.

\subsection{Neural Network Architecture and Training}
Once the synthetic dataset is built, the neural network is trained to learn the forward
pricing map from model parameters to implied volatility surfaces. Following the original
authors' design, we use a fully connected feed-forward neural network with ELU activation
functions. Specifically, our architecture consists of an input layer of 12 nodes (one per
model parameter), 4 hidden layers of 30 neurons each, followed by a linear output layer of
99 nodes, one for each point on the $9 \times 11$ implied volatility surface. The total
number of trainable weights and biases in this network is $\mathbf{6{,}249}$.

We then apply the same normalisations as the authors. The 12 input parameters are first normalised to the interval $[-1, 1]$ using the known
lower and upper bounds of each parameter via the affine map
\[
    \tilde{x}_i \;=\; \frac{x_i - \tfrac{1}{2}(u_i + l_i)}{\tfrac{1}{2}(u_i - l_i)},
\]
where $l_i$ and $u_i$ denote the lower and upper bounds of parameter $i$ respectively.
The 99 output implied volatilities are standardised to zero mean and unit variance using a
\texttt{StandardScaler} fitted exclusively on the training set and subsequently applied to
the test set, preventing any data leakage.

The dataset of 60{,}000 samples is split into a training set of 85\% and a test set of 15\%,
corresponding respectively to 51{,}000 and 9{,}000 parameter configurations.

\medskip
A key adaptation we introduced concerns the loss function used during training. Rather than
using the standard Mean Squared Error (MSE) uniformly across all output points, we designed
a custom \emph{maturity-weighted} loss function. For each output point corresponding to
maturity $T_i$, we assign a scalar weight $w_i$ defined as:
\begin{equation}
    w_i \;=\; \begin{cases} 2 - T_i & \text{if } T_i < 1, \\ 1 & \text{if } T_i \geq 1.
    \end{cases}
\end{equation}

This choice is motivated by two complementary considerations. First, from a market
perspective, short-dated FX options are more actively traded and more sensitive to model
parameters, making accurate reproduction of the near-term smile the main objective
for calibration. Second, and more fundamentally, the short-maturity tenors in our
grid suffer from a discretisation problem. Indeed, because
our Monte Carlo engine uses a single global time grid of $N_{\text{steps}} = 500$ steps per
year over the horizon $T_{\max} = 1$ year, the time step is fixed at
\[
    \Delta t \;=\; \frac{1}{500} \;\approx\; 0.002 \text{ years} \;\approx\; 0.5
    \text{ trading days.}
\]
The shortest maturity in our grid, $T = 0.019$ (approximately one week), is therefore
resolved by only
\[
    n_{T} \;=\; \operatorname{round}(0.019 \times 500) \;=\; \mathbf{10}
    \text{ time steps.}
\]
Ten steps over a one-week horizon is a very coarse discretisation of the rough volatility
process, whose Volterra kernel decays steeply near zero for small Hurst exponents
$H \in (0.025, 0.5)$. This induces a systematic upward bias in the simulated variance
process at short horizons, which propagates into the training labels as simulation noise
concentrated precisely on the short-maturity slice. Assigning higher loss weights to these
tenors does not eliminate the noise, but it forces the network to prioritise fitting through
it rather than ignoring the short end in favour of the better-resolved longer maturities. The training objective therefore becomes:
\begin{equation}
    \hat{w} = \arg\min_{w} \sum_{u=1}^{N_{\mathrm{Train}}} \sum_{i=1}^{n} \sum_{j=1}^{m} w_i \cdot \left(F(\theta_u, w)_{ij} - F^*(\theta_u)_{ij}\right)^2
\end{equation}

The network is trained using the Adam optimiser with a batch size of 32, for a maximum of 200 epochs, with early stopping applied with a patience of 25 epochs, halting training when the validation loss fails to improve, in order to prevent overfitting.

\subsection{Validation on Synthetic Data}

Before applying the framework to real market data, we first validated it on the synthetic test set. This controlled experiment allows us to assess how well the calibration procedure recovers known model parameters when the ground truth is available. For each observation in the test set, we treat the corresponding implied volatility surface as if it were an observed market surface, and run the Levenberg-Marquardt optimiser to find the parameter vector that minimises the distance between the network's output and the target surface. Formally, we solve:
\begin{equation}
    \hat{\theta} = \arg\min_{\theta \in \Theta} \sum_{i=1}^{n} \sum_{j=1}^{m} \left(\tilde{F}(\theta)_{ij} - \sigma_{BS}^{\mathrm{target}}(T_i, k_j)\right)^2
\end{equation}
where $\sigma_{BS}^{\mathrm{target}}$ is the implied volatility surface generated by the Monte Carlo pricer for a known parameter set. The Levenberg-Marquardt algorithm uses the exact analytical Jacobian of the neural network output with respect to its inputs, computed via backpropagation through the network weights, which ensures fast and accurate gradient information. The calibration is performed entirely in the normalised parameter space $[-1, 1]^{12}$, and the recovered parameters are subsequently mapped back to their physical values. This experiment confirmed that the framework generalises correctly to unseen parameter combinations, with calibration errors consistent with the network's approximation accuracy on the test set. The neural network prediction against the Monte Carlo ground truth for a representative out-of-sample configuration is shown in Figure~\ref{fig:test_mc} (Appendix~\ref{app:figures}).

\subsection{Calibration on CME EUR/USD Market Data}

The final and most important test of our implementation is the calibration of the rough Bergomi model to real EUR/USD foreign exchange option data. Our dataset consists of daily implied volatility quotes from CME EUR/USD options, covering the period from start date = 2021-06-08 to end date = 2026-04-13. The data is provided in the standard FX market convention: for each tenor and each of five delta levels (10$\Delta$, 25$\Delta$, 50$\Delta$, 75$\Delta$, 90$\Delta$), we observe a mid implied volatility quote. This gives 45 observed volatility points per trading day, spread across the 9 tenors of our maturity grid.
 
Since the market quotes are expressed in terms of option delta rather than strike price, a conversion step is required before calibration. We convert each delta-volatility pair $(\Delta, \sigma, T)$ to a moneyness value $K/F$ using the Garman-Kohlhagen inversion formula, which is the standard convention in FX markets:
\begin{equation}
    k = \ln\!\left(\frac{K}{F}\right) = -\sigma\sqrt{T}\,\Phi^{-1}(\Delta) + \frac{1}{2}\sigma^2 T
\end{equation}
where $k$ is the log-moneyness, $\Phi^{-1}$ is the inverse of the standard normal cumulative distribution function, and $F$ is the forward price. This produces, for each trading day, a set of 45 scattered $(T, K/F, \sigma)$ observations spread irregularly across the maturity-moneyness space. Since the neural network expects a regular grid of 99 points aligned with our fixed $z$-score grid, we interpolate the 45 market observations onto the target grid using linear interpolation, with nearest-neighbour fallback for any grid points that fall outside the convex hull of the observations. A binary weight vector $w \in \{0, 1\}^{99}$ is constructed to flag the grid points that were covered by linear interpolation, so that the calibration optimiser only fits the network to reliably interpolated points and ignores extrapolated ones.
 
The calibration is then run for each trading day in the dataset using the Levenberg-Marquardt algorithm, with the calibrated parameters from the previous day used as a warm start for the current day. This warm-start strategy significantly reduces the number of iterations needed to converge and produces smoother time series of calibrated parameters. The quality of the fit is evaluated daily using the Root Mean Square Error (RMSE) between the network's predicted volatility surface and the interpolated market surface, computed only over the valid interpolation points.

A representative calibration result on a single trading day is shown in Figure~\ref{fig:calib_eurusd} (Appendix~\ref{app:figures}). The evolution of the global calibration error over the full sample period is displayed in Figure~\ref{fig:error_evolution} (Appendix~\ref{app:figures}).

The calibration error remains below $0.50\%$ for most of the sample period. The two
visible stress episodes correspond to well-identified market events: the 2022--2023 spike
coincides with the most aggressive Fed tightening cycle in four decades, the war in Ukraine
and the ensuing energy crisis that drove EUR/USD below parity for the first time since 2002;
the early 2025 episode reflects the acute market stress triggered by the announcement of
sweeping US tariffs on 2 April 2025, which caused the VIX to surge beyond 50, prompted an
unusual simultaneous sell-off of US Treasuries and the dollar, and generated sharp,
disorderly repricing across FX volatility surfaces. During both periods, the implied
volatility surface exhibits steep skews and elevated wing premiums that fall outside the
parameter ranges of our synthetic training set, explaining the temporary degradation in fit
quality.


\section{Limitations and Possible Improvements}

Despite its many contributions, the framework proposed by Horvath, Muguruza, and Tomas has several limitations.
 
A first limitation is on the dependence on the quality of the training data. The neural network learns solely from synthetic prices generated by Monte Carlo simulation. This means that any error or bias present in the Monte Carlo benchmark is automatically inherited by the network. Thus, the neural network can never be more accurate than the numerical method used to produce its training data. A possible improvement would be to incorporate variance reduction techniques, during data generation, in order to produce higher-quality training samples at a lower computational cost.
 
A second limitation relates to the fixed grid structure. The network is trained to output implied volatilities on a predefined set of maturities and strikes. If a practitioner needs to price an option whose maturity or strike falls outside this grid, the network cannot directly provide an answer and a separate interpolation step is required. While the authors acknowledge this and suggest using standard arbitrage-free interpolation methods, this additional step introduces extra complexity.

Finally, the authors explicitly acknowledge that the paper lacks a formal uncertainty quantification of the calibration procedure. When the network outputs a set of calibrated parameters, there is no measure of how confident we should be in those estimates, nor any systematic way to quantify the sensitivity of the result to small changes in the market data. The authors suggest incorporating a Bayesian analysis of the calibration as a direction for future work, such as \textit{Bayesian neural networks} or \textit{conformal prediction} methods.

% =================================================
\newpage
\begin{appendices}

% -------------------------------------------------
\section{Figures}
\label{app:figures}

\begin{figure}[H]
    \centering
        \includegraphics[width=\textwidth]{test_image_predict.png}
        \caption{Neural network prediction vs.\ Monte Carlo ground truth for a randomly
        drawn out-of-sample parameter configuration (sample \#113):
        $H = 0.1875$, $\nu = 1.9260$, $\rho = 0.2347$,
        $\xi = [0.0054,\ 0.0025,\ 0.0040,\ 0.0048,\ 0.0087,\ 0.0042,\ 0.0043,\
        0.0027,\ 0.0041]$.}
        \label{fig:test_mc}
\end{figure}

\begin{figure}[H]
        \centering
        \includegraphics[width=\textwidth]{Test_fx_good_day.png}
        \caption{Calibration of the Rough Bergomi model to EUR/USD market data on
        14 February 2025. Calibrated parameters:
        $H = 0.1563$, $\nu = 1.6757$, $\rho = -0.1599$,
        $\xi(0.019) = 0.0056$, $\xi(0.038) = 0.0083$, $\xi(0.057) = 0.0058$,
        $\xi(0.082) = 0.0061$, $\xi(0.164) = 0.0062$, $\xi(0.246) = 0.0061$,
        $\xi(0.493) = 0.0064$, $\xi(0.739) = 0.0063$, $\xi(1.000) = 0.0010$.}
        \label{fig:calib_eurusd}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{evolution_timeline.png}
    \caption{Evolution of the global calibration error (RMSE and MAE) over the
    EUR/USD implied volatility surface from 2021 to early 2026.}
    \label{fig:error_evolution}
\end{figure}
\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{surface_vol.png}
    \caption{Visual representaion of the volatility surface}
    \label{fig:placeholder}
\end{figure}
\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{rmse_mat_fx.png}
    \caption{Visualisation of the impact of the custom loss function}
\end{figure}

% -------------------------------------------------
\section{Delta Quoting and the Adaptive Strike Grid}
\label{app:delta_grid}

To demonstrate mathematically why Delta-quoted data inherently forms an adaptive grid, consider the Black-Scholes delta for a call option. Ignoring domestic and foreign interest rate differentials for simplicity, the delta $\Delta$ is given by:
\[
    \Delta = \Phi(d_1) = \Phi\!\left( \frac{\ln(S_0/K) + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}} \right)
\]
where $\Phi$ is the cumulative distribution function of the standard normal distribution. Defining the log-moneyness as $k = \ln(K/S_0)$, substituting $-k$ for $\ln(S_0/K)$, and inverting to solve for $k$ yields:
\[
    \Phi^{-1}(\Delta) = \frac{-k + \frac{1}{2}\sigma^2 T}{\sigma \sqrt{T}}
\]
\[
    k = -\sigma\sqrt{T}\,\Phi^{-1}(\Delta) + \frac{1}{2}\sigma^2 T
\]
The dominant term in this expression is $-\sigma\sqrt{T}\,\Phi^{-1}(\Delta)$. By identifying the inverse normal mapping of the delta as the spatial multiplier $z = -\Phi^{-1}(\Delta)$, we recover the exact functional form of the synthetic strike grid:
\[
    k \;\approx\; z \cdot \sigma \cdot \sqrt{T}
\]
Consequently, the FX market's convention of quoting mid-volatilities at fixed deltas naturally enforces an expansion and contraction of the absolute strike space with maturity. By training the neural network on the synthetic $z_j$ grid, we exactly replicate the topological structure of the observable market, guaranteeing that the network processes realistic, densely informative data without ever needing to extrapolate into extreme, illiquid regions.

\end{appendices}

\end{document}
