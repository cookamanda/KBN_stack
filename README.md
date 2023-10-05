# KBN_stack

## Bayesian formalism for source rate constraints from multiple observations
Assume we want to stack the information from $n$ X-ray observations at the time of radio bursts. Still using \cite{kbn} Bayesian formalism, but construct the posterior probability using bayes rule assuming $n$ observations of $N_i$ X-ray photons for $(i \in 1, \ldots n)$, and average background rates of $B_i$ for $(i \in 1, \ldots n)$, to estimate a rate $S$ for X-ray emission at the time of radio bursts. We assume that $S$ is constant for all radio bursts. Starting from Bayes rule

```math
 f(S| N_1,B_1, \ldots, N_{n}, B_n)  =\frac{P( N_1, B_1, \ldots, N_{n}, B_{n}|S)p(S)}{P(N_1, B_1, \ldots, N_n, B_n)}
```
```math
\begin{align}
f(S| N_1,B_1, \ldots, N_{n}, B_n)  & =\frac{P( N_1, B_1, \ldots, N_{n}, B_{n}|S)p(S)}{P(N_1, B_1, \ldots, N_n, B_n)}\\
   & = \frac{p(S)\prod_{i=1}^{n} \text{Pois}(N_i | \lambda = B_i + S)}{\int_0^\infty\prod_{i=1}^{n} \text{Pois}(N_i | \lambda = B_i + S) d S
\end{align}
```
where $\text{Pois}(k| \lambda)$ is the Poisson probability mass function of $k \in \mathbb{N}$ observed counts given Poisson rate $\lambda \in \mathbb{R}^+$. \cite{kbn} assume a uniform prior for all positive values. We use previous limits placed on the X-ray emission at the time of FRBs to set a conservative but still informative prior. Only during the \nicer\, observations do we have multiple radio bursts. Just one radio burst was detected during the XMM, observations, and hence we use the original method from \cite{kbn} to estimate the posterior on source rate. Since these data are independent, we can use the posterior on source count rate from the \xmm\, observations as our prior for the \nicer\, observations. In order to construct the 95\% credible interval from this equation, again following \cite{kbn}, we enforce that $S_{\text{max}}-S_{\text{min}}$ is minimized and the peak value of the posterior is included. 

## Bayesian formalism for $\eta$ constraints from multiple observations

Previously, we assume a single source rate and estsimate the posterior on that source rate combining information from multiple observations. If one expects that the X-ray fluence from the source should be proportional to the radio fluence, it is desirable to estimate the posterior on the X-ray to radio source fluence, $\eta$, assumed to be constant for all bursts. Again, we define $N_i$ as the number of X-ray photons at the time of each radio burst, $B_i$ is average background rate of X-ray photons at the time of each radio burst and $F_i$ is the calculated radio fluence, with associated error $\sigma_{F_i}$. What is the credible interval on $\eta$ for these multiple observations? In the following derivation, we use $N_i, B_i, F_i$ as shorthand for the more conventional general list $N_0, B_0, F_0, \ldots, N_n, B_n, F_n$ where $n$ is the total number of detected bursts. Starting from Bayes rule, we can write
\begin{align}
    f_{N_i, B_i, F_i}(\eta) & = \frac{p(N_i, B_i,  F_i |\eta)p(\eta)}{C}
\end{align}
 where $C\in \mathbb{R}$ is some normalization constant. To compute the probability density of the observed counts, we must invoke an X-ray rate parameter for each observation, $S_i$, however this value is not known. Instead, we introduce $S_i$ in the following equation via the probability density identity $p(B) = \int_A p(A,B) d A$. 
 
\begin{align}
    p(N_i, B_i, F_i |\eta) & = \int_{S_i} p(N_i, B_i, F_i, S_i | \eta) d S_i \\ 
    & = \int_{S_i}p(N_i, B_i, F_i| S_i) p(S_i | \eta ) \dd S_i 
\end{align}

thus, for $\mathcal{N}_0^{\infty}(\mu, \sigma)$ the $[0,\infty)$ truncated normal distribution centered at $\mu$ with standard deviation $\sigma$, (we introduce the truncation as negative source counts are not physical). 
\begin{align}
    f_{N_i, B_i, F_i}(\eta) & = \frac{1}{C} p(\eta) \int_{S_1}\int_{S_2} \ldots \int_{S_n} p(N_i, B_i,  F_i| \mathcal{S}_i) p(\mathcal{S}_i | \eta ) \dd \mathcal{S}_1\dd \mathcal{S}_2\ldots \dd \mathcal{S}_n\\
    & = \frac{1}{C} p(\eta)\int_{S_1}\int_{S_2} \ldots \int_{S_n} \prod_i \text{Pois}(N_i, B_i, F_i| \lambda = \mathcal{S}_i) \hspace{1.5mm} \mathcal{N}_0^{\infty}\left(\frac{\eta F_\text{i}}{(\text{Flux}/S)},  \frac{\eta}{\text{Flux}/S}\sigma_{F_i}\right) \dd \mathcal{S}_1\dd \mathcal{S}_2\ldots \dd \mathcal{S}_n\\
    & = \frac{1}{C} p(\eta) \prod_i \int_{S_i} \text{Pois}(N_i, B_i, F_i| \lambda = \mathcal{S}_i)  \hspace{1.5mm} \mathcal{N}_0^{\infty}\left(\frac{\eta F_\text{i}}{\text{Flux}/S}, \frac{\eta}{\text{Flux}/S} \sigma_{F_i}\right) \dd \mathcal{S}_i
\end{align}
As above, we will use the the $\eta$ prior implied from the \xmm\, radio burst \cite{kbn} $S$ posterior. In order to compute $\text{Flux}/S$, we will use pimms \citep{1993Legac...3...21M} to convert 1 count/s to X-ray flux for the \nicer\, telescope, given the X-ray absorption \citep[$N_H = 1.42\times 10^{21} \text{cm}^{-2}$;][]{2016AA...594A.116H} along the line of sight.
