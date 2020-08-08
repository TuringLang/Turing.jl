---
title: Variational Inference
toc: true
---

# Overview

In this post we'll have a look at what's know as **variational inference (VI)**, a family of _approximate_ Bayesian inference methods. In particular, we will focus on one of the more standard VI methods called **Automatic Differentation Variational Inference (ADVI)**. If 

Here we'll have a look at the theory behind VI, but if you're interested in how to use ADVI in Turing.jl, [checkout this tutorial](../../tutorials/9-variationalinference).

# Motivation

In Bayesian inference one usually specifies a model as follows: given data \$\$\\{ x\_i\\}\_{i = 1}^n\$\$, 

\$\$
\\begin{align*}
  \text\{prior:     \} \quad z &\sim p(z)   \\\\
  \text\{likelihood:\} \\quad x_i &\overset{\text{i.i.d.}}{\sim} p(x \mid z) \quad  \text{where} \quad i = 1, \dots, n
\\end{align*}
\$\$

where \$\$\overset{\text{i.i.d.}}{\sim}\$\$ denotes that the samples are identically independently distributed. Our goal in Bayesian inference is then to find the _posterior_
\$\$
p(z \mid \\{ x\_i \\}\_{i = 1}^n) = \prod\_{i=1}^{n} p(z \mid x\_i)
\$\$
In general one cannot obtain a closed form expression for \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$, but one might still be able to _sample_ from \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ with guarantees of converging to the target posterior \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ as the number of samples go to \$\$\infty\$\$, e.g. MCMC.

As you are hopefully already aware, Turing.jl provides a lot of different methods with asymptotic exactness guarantees that we can apply to such a problem!

Unfortunately, these unbiased samplers can be prohibitively expensive to run. As the model \$\$p\$\$ increases in complexity, the convergence of these unbiased samplers can slow down dramatically. Still, in the _infinite_ limit, these methods should converge to the true posterior! But infinity is fairly large, like, _at least_ more than 12, so this might take a while. 

In such a case it might be desirable to sacrifice some of these asymptotic guarantees, and instead _approximate_ the posterior \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ using some other model which we'll denote \$\$q(z)\$\$.

There are multiple approaches to take in this case, one of which is **variational inference (VI)**.

# Variational Inference (VI)

In VI, we're looking to approximate \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n )\$\$ using some _approximate_ or _variational_ posterior \$\$q(z)\$\$.

To approximate something you need a notion of what "close" means. In the context of probability densities a standard such "measure" of closeness is the _Kullback-Leibler (KL) divergence_ , though this is far from the only one. The KL-divergence is defined between two densities \$\$q(z)\$\$ and \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ as

\$\$
\begin{align*}
  \mathrm{D\_{KL}} \big( q(z), p(z \mid \\{ x\_i \\}\_{i = 1}^n) \big) &= \int \log \bigg( \frac{q(z)}{\prod\_{i = 1}^n p(z \mid x\_i)} \bigg) q(z) \mathrm{d}{z} \\\\
  &= \mathbb{E}\_{z \sim q(z)} \big[ \log q(z) - \sum\_{i = 1}^n \log p(z \mid x\_i) \big] \\\\
  &= \mathbb{E}\_{z \sim q(z)} \big[ \log q(z) \big] - \sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(z \mid x\_i) \big]
\end{align*}
\$\$


It's worth noting that unfortunately the KL-divergence is _not_ a metric/distance in the analysis-sense due to its lack of symmetry. On the other hand, it turns out that minimizing the KL-divergence that it's actually equivalent to maximizing the log-likelihood! Also, under reasonable restrictions on the densities at hand,

\$\$
\mathrm{D\_{KL}}\big(q(z), p(z \mid \\{ x\_i \\}\_{i = 1}^n) \big) = 0 \quad \iff \quad q(z) = p(z \mid \\{ x\_i \\}\_{i = 1}^n), \quad \forall z
\$\$

Therefore one could (and we will) attempt to approximate \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ using a density \$\$q(z)\$\$ by minimizing the KL-divergence between these two!

One can also show that \$\$\mathrm{D\_{KL}} \ge 0\$\$, which we'll need later. Finally notice that the KL-divergence is only well-defined when in fact \$\$q(z)\$\$ is zero everywhere \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ is zero, i.e.

\$\$
\mathrm{supp}\big(q(z)\big) \subseteq \mathrm{supp}\big(p(z \mid x)\big)
\$\$

Otherwise there might be a point \$\$z\_0 \sim q(z)\$\$ such that \$\$p(z\_0 \mid \\{ x\_i \\}\_{i = 1}^n) = 0\$\$, resulting in \$\$\log\big(\frac{q(z)}{0}\big)\$\$ which doesn't make sense!

One major problem: as we can see in the definition of the KL-divergence, we need \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ for any \$\$z\$\$ if we want to compute the KL-divergence between this and \$\$q(z)\$\$. We don't have that. The entire reason we even do Bayesian inference is that we don't know the posterior! Cleary this isn't going to work. _Or is it?!_

## Computing KL-divergence without knowing the posterior

First off, recall that

\$\$
p(z \mid x\_i) = \frac{p(x\_i, z)}{p(x\_i)}
\$\$

so we can write

\$\$
\begin{align*}
\mathrm{D\_{KL}} \big( q(z), p(z \mid \\{ x\_i \\}\_{i = 1}^n) \big) &= \mathbb{E}\_{z \sim q(z)} \big[ \log q(z) \big] - \sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i, z) - \log p(x\_i) \big] \\\\
    &= \mathbb{E}\_{z \sim q(z)} \big[ \log q(z) \big] - \sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i, z) \big] + \sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i) \big] \\\\ 
    &= \mathbb{E}\_{z \sim q(z)} \big[ \log q(z) \big] - \sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i, z) \big] + \sum\_{i = 1}^n \log p(x\_i)
\end{align*}
\$\$

where in the last equality we used the fact that \$\$p(x\_i)\$\$ is independent of \$\$z\$\$.

Now you're probably thinking "Oh great! Now you've introduced \$\$p(x\_i)\$\$ which we _also_ can't compute (in general)!". Woah. Calm down human. Let's do some more algebra. The above expression can be rearranged to

\$\$
\mathrm{D\_{KL}} \big( q(z), p(z \mid \\{ x\_i \\}\_{i = 1}^n) \big) + \underbrace{\sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i, z) \big] - \mathbb{E}\_{z \sim q(z)} \big[ \log q(z) \big]}\_{=: \mathrm{ELBO}(q)} = \underbrace{\sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i) \big]}\_{\text{constant}}
\$\$

See? The left-hand side is _constant_ and, as we mentioned before, \$\$\mathrm{D\_{KL}} \ge 0\$\$. What happens if we try to _maximize_ the term we just gave the completely arbitrary name \$\$\mathrm{ELBO}\$\$? Well, if \$\$\mathrm{ELBO}\$\$ goes up while \$\$p(x\_i)\$\$ stays constant then \$\$\mathrm{D\_{KL}}\$\$ _has to_ go down! That is, the \$\$q(z)\$\$ which _minimizes_ the KL-divergence is the same \$\$q(z)\$\$ which _maximizes_ \$\$\mathrm{ELBO}(q)\$\$:

\$\$
\underset{q}{\mathrm{argmin}} \ \mathrm{D\_{KL}} \big( q(z), p(z \mid \\{ x\_i \\}\_{i = 1}^n) \big) = \underset{q}{\mathrm{argmax}} \ \mathrm{ELBO}(q)
\$\$

where

\$\$
\begin{align*}
\mathrm{ELBO}(q) &:= \bigg( \sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i, z) \big]  \bigg) - \mathbb{E}\_{z \sim q(z)} \big[ \log q(z) \big] \\\\
    &= \bigg( \sum\_{i = 1}^n \mathbb{E}\_{z \sim q(z)} \big[ \log p(x\_i, z) \big] \bigg) + \mathbb{H}\big( q(z) \big)
\end{align*}
\$\$

and \$\$\mathbb{H} \big(q(z) \big)\$\$ denotes the [(differential) entropy](https://www.wikiwand.com/en/Differential_entropy) of \$\$q(z)\$\$.

Assuming joint \$\$p(x\_i, z)\$\$ and the entropy \$\$\mathbb{H}\big(q(z)\big)\$\$ are both tractable, we can use a Monte-Carlo for the remaining expectation. This leaves us with the following tractable expression

\$\$
\underset{q}{\mathrm{argmin}} \ \mathrm{D\_{KL}} \big( q(z), p(z \mid \\{ x\_i \\}\_{i = 1}^n) \big) \approx \underset{q}{\mathrm{argmax}} \ \widehat{\mathrm{ELBO}}(q)
\$\$

where

\$\$
\widehat{\mathrm{ELBO}}(q) = \frac{1}{m} \bigg( \sum\_{k = 1}^m \sum\_{i = 1}^n \log p(x\_i, z\_k) \bigg) + \mathbb{H} \big(q(z)\big) \quad \text{where} \quad z\_k \sim q(z) \quad \forall k = 1, \dots, m
\$\$

Hence, as long as we can sample from \$\$q(z)\$\$ somewhat efficiently, we can indeed minimize the KL-divergence! Neat, eh?

Sidenote: in the case where \$\$q(z)\$\$ is tractable but \$\$\mathbb{H} \big(q(z) \big)\$\$ is _not_ , we can use an Monte-Carlo estimate for this term too but this generally results in a higher-variance estimate.

Also, I fooled you real good: the ELBO _isn't_ an arbitrary name, hah! In fact it's an abbreviation for the **expected lower bound (ELBO)** because it, uhmm, well, it's the _expected_ lower bound (remember \$\$\mathrm{D\_{KL}} \ge 0\$\$). Yup.

## Maximizing the ELBO

Finding the optimal \$\$q\$\$ over _all_ possible densities of course isn't feasible. Instead we consider a family of _parameterized_ densities \$\$\mathscr{D}\_{\Theta}\$\$ where \$\$\Theta\$\$ denotes the space of possible parameters. Each density in this family \$\$q\_{\theta} \in \mathscr{D}\_{\Theta}\$\$ is parameterized by a unique \$\$\theta \in \Theta\$\$. Moreover, we'll assume
1. \$\$q\_{\theta}(z)\$\$, i.e. evaluating the probability density \$\$q\$\$ at any point \$\$z\$\$, is differentiable
2. \$\$z \sim q\_{\theta}(z)\$\$, i.e. the process of sampling from \$\$q\_{\theta}(z)\$\$, is differentiable

(1) is fairly straight-forward, but (2) is a bit tricky. What does it even mean for a _sampling process_ to be differentiable? This is quite an interesting problem in its own right and would require something like a [50-page paper to properly review the different approaches (highly recommended read)](https://arxiv.org/abs/1906.10652).

We're going to make use of a particular such approach which goes under a bunch of different names: _reparametrization trick_, _path derivative_, etc. This refers to making the assumption that all elements \$\$q\_{\theta} \in \mathscr{Q}\_{\Theta}\$\$ can be considered as reparameterizations of some base density, say \$\$\bar{q}(z)\$\$. That is, if \$\$q\_{\theta} \in \mathscr{Q}\_{\Theta}\$\$ then

\$\$
z \sim q\_{\theta}(z) \quad \iff \quad z := g\_{\theta}(\tilde{z}) \quad \text{where} \quad \bar{z} \sim \bar{q}(z)
\$\$

for some function \$\$g\_{\theta}\$\$ differentiable wrt. \$\$\theta\$\$. So all \$\$q\_{\theta} \in \mathscr{Q}\_{\Theta}\$\$ are using the *same* reparameterization-function \$\$g\$\$ but each \$\$q\_{\theta}\$\$ correspond to different choices of \$\$\theta\$\$ for \$\$f\_{\theta}\$\$.

Under this assumption we can differentiate the sampling process by taking the derivative of \$\$g\_{\theta}\$\$ wrt. \$\$\theta\$\$, and thus we can differentiate the entire \$\$\widehat{\mathrm{ELBO}}(q\_{\theta})\$\$ wrt. \$\$\theta\$\$! With the gradient available we can either try to solve for optimality either by setting the gradient equal to zero or maximize \$\$\widehat{\mathrm{ELBO}}(q\_{\theta})\$\$ stepwise by traversing \$\$\mathscr{Q}\_{\Theta}\$\$ in the direction of steepest ascent. For the sake of generality, we're going to go with the stepwise approach.

With all this nailed down, we eventually reach the section on **Automatic Differentiation Variational Inference (ADVI)**.

## Automatic Differentiation Variational Inference (ADVI)

So let's revisit the assumptions we've made at this point:
1. The variational posterior \$\$q\_{\theta}\$\$ is in a parameterized family of densities denoted \$\$\mathscr{Q}\_{\Theta}\$\$, with \$\$\theta \in \Theta\$\$.
2. \$\$\mathscr{Q}\_{\Theta}\$\$ is a space of _reparameterizable_ densities with \$\$\bar{q}(z)\$\$ as the base-density.
3. The parameterization function \$\$g\_{\theta}\$\$ is differentiable wrt. \$\$\theta\$\$.
4. Evaluation of the probability density \$\$q\_{\theta}(z)\$\$ is differentiable wrt. \$\$\theta\$\$.
5. \$\$\mathbb{H}\big(q\_{\theta}(z)\big)\$\$ is tractable.
6. Evaluation of the joint density \$\$p(x, z)\$\$ is tractable and differentiable wrt. \$\$z\$\$
7. The support of \$\$p(z \mid x)\$\$ is a subspace of the support of \$\$q(z)\$\$: \$\$\mathrm{supp}\big(p(z \mid x)\big) \subseteq \mathrm{supp}\big(q(z)\big)\$\$.


All of these are not *necessary* to do VI, but they are very convenient and results in a fairly flexible approach. One distribution which has a density satisfying all of the above assumptions _except_ (7) (we'll get back to this in second) for any tractable and differentiable \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ is the good ole' Gaussian/normal distribution:

\$\$
z \sim \mathcal{N}(\mu, \Sigma) \quad \iff \quad z = g\_{\mu, L}(\bar{z}) := \mu + L^T \tilde{z} \quad \text{where} \quad \bar{z} \sim \bar{q}(z) := \mathcal{N}(1\_d, I\_{d \times d})
\$\$

where \$\$\Sigma = L L^T\$\$, with \$\$L\$\$ obtained from the Cholesky-decomposition. Abusing notation a bit, we're going to write

\$\$
\theta = (\mu, \Sigma) := (\mu\_1, \dots, \mu\_d, L\_{11}, \dots, L\_{1, d}, L\_{2, 1}, \dots, L\_{2, d}, \dots, L\_{d, 1}, \dots, L\_{d, d})
\$\$

With this assumption we finally have a tractable expression for \$\$\widehat{\mathrm{ELBO}}(q\_{\mu, \Sigma})\$\$! Well, assuming (7) is holds. Since a Gaussian has non-zero probability on the entirety of \$\$\mathbb{R}^d\$\$, we also require \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$ to have non-zero probability on all of \$\$\mathbb{R}^d\$\$.

Though not necessary, we'll often make a *mean-field* assumption for the variational posterior \$\$q(z)\$\$, i.e. assume independence between the latent variables. In this case, we'll write

\$\$
\theta = (\mu, \sigma^2) := (\mu\_1, \dots, \mu\_d, \sigma\_1^2, \dots, \sigma_d^2)
\$\$

### Examples
As a (trivial) example we could apply the approach described above to is the following generative model for \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$:

\$\$
\begin{align*}
    m &\sim \mathcal{N}(0, 1) \\\\
    x\_i &\overset{\text{i.i.d.}}{=} \mathcal{N}(m, 1), \quad i = 1, \dots, n
\end{align*}
\$\$

In this case \$\$z = m\$\$ and we have the posterior defined \$\$p(m \mid \\{ x\_i \\}\_{i = 1}^n) = p(m) \prod\_{i = 1}^n p(x\_i \mid m)\$\$. Then the variational posterior would be

\$\$
q\_{\mu, \sigma} = \mathcal{N}(\mu, \sigma^2) \quad \text{where} \quad \mu \in \mathbb{R}, \ \sigma^2 \in \mathbb{R}^{ + }
\$\$

And since prior of \$\$m\$\$, \$\$\mathcal{N}(0, 1)\$\$, has non-zero probability on the entirety of \$\$\mathbb{R}\$\$, same as \$\$q(m)\$\$, i.e. assumption (7) above holds, everything is fine and life is good.

But what about this generative model for \$\$p(z \mid \\{ x\_i \\}\_{i = 1}^n)\$\$:

\$\$
\begin{align*}
    s &\sim \mathrm{InverseGamma}(2, 3) \\\\
    m &\sim \mathcal{N}(0, s) \\\\
    x\_i &\overset{\text{i.i.d.}}{=} \mathcal{N}(m, s), \quad i = 1, \dots, n
\end{align*}
\$\$

with posterior \$\$p(s, m \mid \\{ x\_i \\}\_{i = 1}^n) = p(s) p(m \mid s) \prod\_{i = 1}^n p(x\_i \mid s, m)\$\$ and the mean-field variational posterior \$\$q(s, m)\$\$ will be

\$\$
q\_{\mu\_1, \mu\_2, \sigma_1^2, \sigma_2^2}(s, m) = p\_{\mathcal{N}(\mu\_1, \sigma\_1^2)}(s) p\_{\mathcal{N}(\mu\_2, \sigma_2^2)}(m)
\$\$

where we've denoted the evaluation of the probability density of a Gaussian as \$\$p\_{\mathcal{N}(\mu, \sigma^2)}(x)\$\$.

Observe that \$\$\mathrm{InverseGamma}(2, 3)\$\$ has non-zero probability only on \$\$\mathbb{R}^{ + } := (0, \infty)\$\$ which is clearly not all of \$\$\mathbb{R}\$\$ like \$\$q(s, m)\$\$ has, i.e.

\$\$
\mathrm{supp} \big( q(s, m) \big) \not\subseteq \mathrm{supp} \big( p(z \mid \\{ x\_i \\}\_{i = 1}^n) \big)
\$\$

Recall from the definition of the KL-divergence that when this is the case, the KL-divergence isn't well defined. This gets us to the *automatic* part of ADVI.

### "Automatic"? How?

For a lot of the standard (continuous) densities \$\$p\$\$ we can actually construct a probability density \$\$\tilde{p}\$\$ with non-zero probability on all of \$\$\mathbb{R}\$\$ by *transforming* the "constrained" probability density \$\$p\$\$ to \$\$\tilde{p}\$\$. In fact, in these cases this is a one-to-one relationship. As we'll see, this helps solve the support-issue we've been going on and on about.

#### Transforming densities using change of variables

If we want to compute the probability of \$\$x\$\$ taking a value in some set \$\$A \subseteq \mathrm{supp} \big( p(x) \big)\$\$, we have to integrate \$\$p(x)\$\$ over \$\$A\$\$, i.e.

\$\$
\mathbb{P}\_p(x \in A) = \int\_A p(x) \mathrm{d}x
\$\$

This means that if we have a differentiable bijection \$\$f: \mathrm{supp} \big( q(x) \big) \to \mathbb{R}^d\$\$ with differentiable inverse \$\$f^{-1}: \mathbb{R}^d \to \mathrm{supp} \big( p(x) \big)\$\$, we can perform a change of variables

\$\$
\mathbb{P}\_p(x \in A) = \int\_{f^{-1}(A)} p \big(f^{-1}(y) \big) \ \big| \det \mathcal{J}\_{f^{-1}}(y) \big| \ \mathrm{d}y
\$\$

where \$\$\mathcal{J}\_{f^{-1}}(x)\$\$ denotes the jacobian of \$\$f^{-1}\$\$ evaluted at \$\$x\$\$. Observe that this defines a probability distribution

\$\$
\mathbb{P}\_{\tilde{p}}\big(y \in f^{-1}(A) \big) = \int\_{f^{-1}(A)} \tilde{p}(y) \mathrm{d}y
\$\$

since \$\$f^{-1}\big(\mathrm{supp} (p(x)) \big) = \mathbb{R}^d\$\$ which has probability 1. This probability distribution has *density* \$\$\tilde{p}(y)\$\$ with \$\$\mathrm{supp} \big( \tilde{p}(y) \big) = \mathbb{R}^d\$\$, defined

\$\$
\tilde{p}(y) = p \big( f^{-1}(y) \big) \ \big| \det \mathcal{J}\_{f^{-1}}(y) \big|
\$\$

or equivalently

\$\$
\tilde{p} \big( f(x) \big) = \frac{p(x)}{\big| \det \mathcal{J}\_{f}(x) \big|}
\$\$

due to the fact that

\$\$
\big| \det \mathcal{J}\_{f^{-1}}(y) \big| = \big| \det \mathcal{J}\_{f}(x) \big|^{-1}
\$\$

*Note: it's also necessary that the log-abs-det-jacobian term is non-vanishing. This can for example be accomplished by assuming \$\$f\$\$ to also be elementwise monotonic.*

#### Back to VI

So why is this is useful? Well, we're looking to generalize our approach using a normal distribution to cases where the supports don't match up. How about defining \$\$q(z)\$\$ by

\$\$
\begin{align*}
  \eta &\sim \mathcal{N}(\mu, \Sigma) \\\\
  z &= f^{-1}(\eta)
\end{align*}
\$\$

where \$\$f^{-1}: \mathbb{R}^d \to \mathrm{supp} \big( p(z \mid x) \big)\$\$ is a differentiable bijection with differentiable inverse. Then \$\$z \sim q\_{\mu, \Sigma}(z) \implies z \in \mathrm{supp} \big( p(z \mid x) \big)\$\$ as we wanted. The resulting variational density is

\$\$
q\_{\mu, \Sigma}(z) = p\_{\mathcal{N}(\mu, \Sigma)}\big( f(z) \big) \ \big| \det \mathcal{J}\_{f}(z) \big|
\$\$

Note that the way we've constructed \$\$q(z)\$\$ here is basically a reverse of the approach we described above. Here we sample from a distribution with support on \$\$\mathbb{R}\$\$ and transform *to* \$\$\mathrm{supp} \big( p(z \mid x) \big)\$\$.

If we want to write the ELBO explicitly in terms of \$\$\eta\$\$ rather than \$\$z\$\$, the first term in the ELBO becomes

\$\$
\begin{align*}
  \mathbb{E}\_{z \sim q\_{\mu, \Sigma}(z)} \big[ \log p(x\_i, z) \big] &= \mathbb{E}\_{\eta \sim \mathcal{N}(\mu, \Sigma)} \Bigg[ \log \frac{p\big(x\_i, f^{-1}(\eta) \big)}{\big| \det \mathcal{J}\_{f^{-1}}(\eta) \big|} \Bigg] \\\\
  &= \mathbb{E}\_{\eta \sim \mathcal{N}(\mu, \Sigma)} \big[ \log p\big(x\_i, f^{-1}(\eta) \big) \big] - \mathbb{E}\_{\eta \sim \mathcal{N}(\mu, \Sigma)} \big[ \big| \det \mathcal{J}\_{f^{-1}}(\eta) \big| \big]
\end{align*}
\$\$

The entropy is invariant under change of variables, thus \$\$\mathbb{H} \big(q\_{\mu, \Sigma}(z)\big)\$\$ is simply the entropy of the normal distribution which is known analytically. 

Hence, the resulting empirical estimate of the ELBO is

\$\$
\begin{align*}
\widehat{\mathrm{ELBO}}(q\_{\mu, \Sigma}) &= \frac{1}{m} \bigg( \sum\_{k = 1}^m \sum\_{i = 1}^n \Big(\log p\big(x\_i, f^{-1}(\eta\_k)\big) - \log \big| \det \mathcal{J}\_{f^{-1}}(\eta\_k) \big| \Big) \bigg) + \mathbb{H} \big(p\_{\mathcal{N}(\mu, \Sigma)}(z)\big) \\\\
& \text{where} \quad z\_k  \sim \mathcal{N}(\mu, \Sigma) \quad \forall k = 1, \dots, m
\end{align*}
\$\$

And maximizing this wrt. \$\$\mu\$\$ and \$\$\Sigma\$\$ is what's referred to as **Automatic Differentation Variational Inference (ADVI)**!

Now if you want to try it out, [check out the tutorial on how to use ADVI in Turing.jl](../../tutorials/9-variationalinference)!
