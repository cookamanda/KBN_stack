# KBN_stack

## Bayesian formalism for source rate constraints from multiple observations
We seek to combine information from $n$ X-ray observations at the time of radio bursts, which can be treated as independent trials if sufficiently separated in time (i.e., the time between the FRBs is long compared to the durations tested). We follow the [Kraft et al. (1991)](https://ui.adsabs.harvard.edu/abs/1991ApJ...374..344K/abstract) Bayesian formalism, but construct the posterior probability using Bayes rule assuming $n$ observations of $N_i$ X-ray photons for $(i \in 1, \ldots n)$, and (known) average background rates of $B_i$ for $(i \in 1, \ldots, n)$, to estimate a rate $S$ for X-ray emission at the time of radio bursts. This method assumes that $S$ is constant for all radio bursts. From Bayes rule, we have can derive the posterior $f$ on the rate $S$:
```math
\begin{align}
p(S) & \propto 1\\
N_i & \sim \text{Poisson}(\lambda_i = B_i + S)\\
    f(S| N_1, \ldots, N_{n}) & =\frac{ P( N_1, \ldots, N_{n}|S)p(S)  }{P(N_1 \ldots, N_n)}\\
    & \propto \prod_{i=1}^{n} \text{Poisson}(N_i | \lambda_i = B_i + S), 
\end{align}
```

where $\text{Poisson}(k| \lambda)$ is the Poisson probability mass function of $k \in \mathbb{N}$ observed counts given Poisson rate $\lambda \in \mathbb{R}^+$ and $B_i$ is assumed constant and known for each observation $n$.  [Kraft et al. (1991)](https://ui.adsabs.harvard.edu/abs/1991ApJ...374..344K/abstract) assume an improper uniform prior p(S) = c for all positive values. In order to construct a proper (finite) uniform prior distribution, one should instead assume $p(S) = c \ \forall S \in [0,x]$ for some appropriately large value $x \in \mathbb{R}^+$ of X-ray rate, and practically, this will be enforced by any analytic implementation which computes this posterior. 
Previous limits placed on the X-ray emission at the time of FRBs can instead be used to set a conservative but still informative prior.
In order to construct the 99.7\% credible interval from this equation, again following [Kraft et al. (1991)](https://ui.adsabs.harvard.edu/abs/1991ApJ...374..344K/abstract), we enforce that the difference in the flux bounds of the credible interval ($S_{\text{max}}-S_{\text{min}}$) is minimized and the peak value of the posterior is included. This is known as the highest posterior density interval. 

## Bayesian formalism for η constraints from multiple observations

Above, we assume a single source rate, $S$, and calculate the posterior on that source rate by combining information from multiple observations. If one expects that the X-ray fluence from the source should be proportional to the radio fluence, it is desirable to estimate the posterior on the relative X-ray to radio source fluence, $\eta_{\text{\,x/r}}$. We thus assume here that $\eta_{\text{\,x/r}}$ is constant for all bursts, and that $S_i$ can vary. Again, we define $N_i$ as the number of X-ray photons at the time of each radio burst, $B_i$ is the average background rate of X-ray photons at the time of each radio burst and $F_{\text{radio, }i}$ is the calculated radio fluence, with associated error $\sigma_{F_{\text{radio, }i}}$. What is the credible interval on $\eta_{	ext{\,x/r}}$ for these multiple observations? In the following derivation, we use $N_i, B_i, F_i$ as shorthand for the more conventional general list $N_0, B_0, F_0, \ldots, N_n, B_n, F_n$ where $n$ is the total number of detected bursts. We will compute the posterior of the observations given following model, introduced in Section 3.4 of Cook et al 2024:
```math
\begin{align}
  \text{Level I} &&  N_i & \sim \text{Poisson}(\lambda_i = S_i + B_i)\\
   \text{Level II} && S_i & \sim \mathcal{N}^\infty_0  \left( \frac{\eta_{\text{\,x/r}} F_{\text{radio}, i}}{(\text{Flux/S})} , \frac{\eta_{\text{\,x/r}} \sigma_{F_{\text{radio}, i}}}{(\text{Flux/S})} \right), 
\end{align}
```
where(Flux/S)$\in \mathbb{R}^+$ is a conversion parameter to turn the X-ray count rates into fluxes. This value depends on the underlying spectral model assumed and the effective area of the X-ray telescope, but can be computed using standard X-ray tools. We assume a blackbody spectral model with $kT = 10$\,keV for the bursts and present the corresponding (Flux/S) parameter for each observation in Table 4 of Cook et al 2024.  $\mathcal{N}_0^\infty (\mu, \sigma)$ denotes a normal distribution truncated on the left at $0$ with mean $\mu \in \mathbb{R}$ and standard deviation $\sigma \in \mathbb{R}^+$. The truncation is introduced because negative source counts are not physical. Poisson$(\lambda)$ denotes the Poisson distribution with rate parameter $\lambda \in \mathbb{R}^+$. The $B_i$ are treated as known and fixed, but an additional model for the error can be added in Level II if there are significant uncertainties in this estimation or the background rate is variable. 
Starting again from Bayes rule, we can write the posterior $f(\eta_{	ext{\,x/r}})$
```math
\begin{align}
    f(\eta_{\text{\,x/r}}) & = \frac{p(N_i|\eta_{\text{\,x/r}})p(\eta_{	ext{\,x/r}})}{C},
\end{align}
```
 where $C\in \mathbb{R}$ is some normalization constant. To compute the probability density of the observed counts, we must invoke an X-ray rate parameter for each observation, $S_i$, however this value is not known. Instead, we assume a hierarchical Bayesian model, introducing $S_i$ as a random variable in the following equation using the chain rule of probability through the identity $p(A|B) = \int_C p(A,C|B) d C = \int_C p(C|B) p(A|B,C)d C$ for random variables $A,B,C$:
```math
 \begin{align}
 f(\eta_{\text{\,x/r}}) & = \frac{1}{C} p(\eta_{\text{\,x/r}}) \int_{S_1}\int_{S_2} \ldots \int_{S_n} p(N_i | S_i, \eta_{\text{\,x/r}}) p(S_i | \eta_{\text{\,x/r}}) d S_1d S_2\ldots d S_n\\
    & \propto p(\eta_{\text{\,x/r}}) \int_{S_1}\int_{S_2} \ldots \int_{S_n} p(N_i | S_i,\eta_{	ext{\,x/r}}) p(S_i | \eta_{	ext{\,x/r}}) d S_1d S_2\ldots d S_n\\
    & \propto p(\eta_{\text{\,x/r}})\int_{S_1}\int_{S_2} \ldots \int_{S_n} \prod_i \text{Pois}(N_i | \lambda_i = S_i+B_i) \hspace{1.5mm} \mathcal{N}^\infty_0  \left( \frac{\eta_{\text{\,x/r}} F_{\text{radio}, i}}{(\text{Flux/S})} , \frac{\eta_{\text{\,x/r}} \sigma_{F_{\text{radio}, i}}}{(\text{Flux/S})} \right)d S_1d S_2\ldots d S_n\\
    & \propto p(\eta_{\text{\,x/r}}) \prod_i \int^\infty_{0} 
      \hspace{1.5mm} 
      \frac{(\text{Flux/S})(B_i + S_i)^{N_i}}{ \sqrt{2\pi}\eta_{\text{\,x/r}}\sigma_{F_{\text{radio}, i}} (N_i!)} \exp[-\left(B_i + S_i +\frac{(\text{Flux/S})^2\left(S_i - \frac{\eta_{\text{\,x/r}}F_{\text{radio}, i}}{(\text{Flux/S})}\right)^2}{2\eta_{\text{i \,x/r}}^2 \sigma^2_{F_{\text{radio}, i}}}\right)]
 d S_i\\
     & \propto p(\eta_{\text{\,x/r}}) \exp(-\sum_{i=1}^n  B_i) \prod_{i=1}^{n} \int^\infty_{0} 
      \hspace{1.5mm} 
      \frac{(\text{Flux/S})(B_i + S_i)^{N_i}}{ \sqrt{2\pi}\eta_{	ext{\,x/r}} \sigma_{F_{\text{radio}, i}} (N_i!)} \exp[-\left( S_i +\frac{(\text{Flux/S})^2\left(S_i - \frac{\eta_{\text{\,x/r}} F_{\text{radio}, i}}{(\text{Flux/S})}\right)^2}{2\eta_{\text{i,x/r}}^2 \sigma^2_{F_{\text{radio}, i}}}\right)]
 d S_i\\
 & \propto p(\eta_{	ext{\,x/r}})  \prod_{i=1}^{n} \int^\infty_{0} 
      \hspace{1.5mm} 
      \frac{(B_i + S_i)^{N_i}}{\eta_{\text{\,x/r}} } \exp[-\left( S_i +\frac{(\text{Flux/S})^2\left(S_i - \frac{\eta_{\text{\,x/r}} F_{\text{radio}, i}}{(\text{Flux/S})}\right)^2}{2\eta_{\text{\,x/r}}^2 \sigma^2_{F_{\text{radio}, i}}}\right)]
 d S_i. \label{eqn:etalikelihood}
\end{align}
```
This expression can be numerically integrated directly and normalized, or estimated with MCMC methods. We use the posterior from a previous independent trial as our prior when stacking. The reported credible regions correspond to the highest posterior density interval.  
