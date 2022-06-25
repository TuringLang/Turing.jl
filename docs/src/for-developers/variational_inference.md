---
title: Variational Inference
toc: true
---

# Overview

In this post we'll have a look at what's known as **variational inference (VI)**, a family of _approximate_ Bayesian inference methods. In particular, we will focus on one of the more standard VI methods called **Automatic Differentiation Variational Inference (ADVI)**. 

Here we'll have a look at the theory behind VI, but if you're interested in how to use ADVI in Turing.jl, [check out this tutorial](../../tutorials/09-variational-inference).

# Motivation

In Bayesian inference one usually specifies a model as follows: given data $\\{ x_i\\}_{i = 1}^n$, 

$$
\begin{align*}
  \text{prior:} \quad z &\sim p(z)   \\\\
  \text{likelihood:} \quad x_i &\overset{\text{i.i.d.}}{\sim} p(x \mid z) \quad  \text{where} \quad i = 1, \dots, n
\end{align*}
$$

where $\overset{\text{i.i.d.}}{\sim}$ denotes that the samples are identically independently distributed. Our goal in Bayesian inference is then to find the _posterior_
$$
p(z \mid \{ x_i \}_{i = 1}^n) = \prod_{i=1}^{n} p(z \mid x_i).
$$
In general one cannot obtain a closed form expression for <span>$p(z \mid \\{ x_i \\}\_{i = 1}^n)$</span>, but one might still be able to _sample_ from $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ with guarantees of converging to the target posterior $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ as the number of samples go to $\infty$, e.g. MCMC.

As you are hopefully already aware, Turing.jl provides a lot of different methods with asymptotic exactness guarantees that we can apply to such a problem!

Unfortunately, these unbiased samplers can be prohibitively expensive to run. As the model $p$ increases in complexity, the convergence of these unbiased samplers can slow down dramatically. Still, in the _infinite_ limit, these methods should converge to the true posterior! But infinity is fairly large, like, _at least_ more than 12, so this might take a while. 

In such a case it might be desirable to sacrifice some of these asymptotic guarantees, and instead _approximate_ the posterior $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ using some other model which we'll denote $q(z)$.

There are multiple approaches to take in this case, one of which is **variational inference (VI)**.

# Variational Inference (VI)

In VI, we're looking to approximate $p(z \mid \\{ x_i \\}\_{i = 1}^n )$ using some _approximate_ or _variational_ posterior $q(z)$.

To approximate something you need a notion of what "close" means. In the context of probability densities a standard such "measure" of closeness is the _Kullback-Leibler (KL) divergence_ , though this is far from the only one. The KL-divergence is defined between two densities $q(z)$ and $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ as

$$
\begin{align*}
  \mathrm{D_{KL}} \left( q(z), p(z \mid \{ x_i \}_{i = 1}^n) \right) &= \int \log \left( \frac{q(z)}{\prod_{i = 1}^n p(z \mid x_i)} \right) q(z) \mathrm{d}{z} \\\\
  &= \mathbb{E}_{z \sim q(z)} \left[ \log q(z) - \sum_{i = 1}^n \log p(z \mid x_i) \right] \\\\
  &= \mathbb{E}_{z \sim q(z)} \left[ \log q(z) \right] - \sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(z \mid x_i) \right].
\end{align*}
$$

It's worth noting that unfortunately the KL-divergence is _not_ a metric/distance in the analysis-sense due to its lack of symmetry. On the other hand, it turns out that minimizing the KL-divergence that it's actually equivalent to maximizing the log-likelihood! Also, under reasonable restrictions on the densities at hand,

$$
\mathrm{D_{KL}}\left(q(z), p(z \mid \{ x_i \}_{i = 1}^n) \right) = 0 \quad \iff \quad q(z) = p(z \mid \{ x_i \}_{i = 1}^n), \quad \forall z.
$$

Therefore one could (and we will) attempt to approximate $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ using a density $q(z)$ by minimizing the KL-divergence between these two!

One can also show that $\mathrm{D_{KL}} \ge 0$, which we'll need later. Finally notice that the KL-divergence is only well-defined when in fact $q(z)$ is zero everywhere $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ is zero, i.e.

$$
\mathrm{supp}\left(q(z)\right) \subseteq \mathrm{supp}\left(p(z \mid x)\right).
$$

Otherwise, there might be a point $z_0 \sim q(z)$ such that $p(z_0 \mid \\{ x_i \\}\_{i = 1}^n) = 0$, resulting in $\log\left(\frac{q(z)}{0}\right)$ which doesn't make sense!

One major problem: as we can see in the definition of the KL-divergence, we need $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ for any $z$ if we want to compute the KL-divergence between this and $q(z)$. We don't have that. The entire reason we even do Bayesian inference is that we don't know the posterior! Cleary this isn't going to work. _Or is it?!_

## Computing KL-divergence without knowing the posterior

First off, recall that

$$
p(z \mid x_i) = \frac{p(x_i, z)}{p(x_i)}
$$

so we can write

$$
\begin{align*}
\mathrm{D_{KL}} \left( q(z), p(z \mid \{ x_i \}_{i = 1}^n) \right) &= \mathbb{E}_{z \sim q(z)} \left[ \log q(z) \right] - \sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i, z) - \log p(x_i) \right] \\\\
    &= \mathbb{E}_{z \sim q(z)} \left[ \log q(z) \right] - \sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i, z) \right] + \sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i) \right] \\\\ 
    &= \mathbb{E}_{z \sim q(z)} \left[ \log q(z) \right] - \sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i, z) \right] + \sum_{i = 1}^n \log p(x_i),
\end{align*}
$$

where in the last equality we used the fact that $p(x_i)$ is independent of $z$.

Now you're probably thinking "Oh great! Now you've introduced $p(x_i)$ which we _also_ can't compute (in general)!". Woah. Calm down human. Let's do some more algebra. The above expression can be rearranged to

$$
\mathrm{D_{KL}} \left( q(z), p(z \mid \{ x_i \}_{i = 1}^n) \right) + \underbrace{\sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i, z) \right] - \mathbb{E}_{z \sim q(z)} \left[ \log q(z) \right]}_{=: \mathrm{ELBO}(q)} = \underbrace{\sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i) \right]}_{\text{constant}}.
$$

See? The left-hand side is _constant_ and, as we mentioned before, $\mathrm{D\_{KL}} \ge 0$. What happens if we try to _maximize_ the term we just gave the completely arbitrary name $\mathrm{ELBO}$? Well, if $\mathrm{ELBO}$ goes up while $p(x_i)$ stays constant then $\mathrm{D\_{KL}}$ _has to_ go down! That is, the $q(z)$ which _minimizes_ the KL-divergence is the same $q(z)$ which _maximizes_ $\mathrm{ELBO}(q)$:

$$
\underset{q}{\mathrm{argmin}} \  \mathrm{D_{KL}} \left( q(z), p(z \mid \{ x_i \}_{i = 1}^n) \right) = \underset{q}{\mathrm{argmax}} \ \mathrm{ELBO}(q)
$$

where

$$
\begin{align*}
\mathrm{ELBO}(q) &:= \left( \sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i, z) \right]  \right) - \mathbb{E}_{z \sim q(z)} \left[ \log q(z) \right] \\\\
    &= \left( \sum_{i = 1}^n \mathbb{E}_{z \sim q(z)} \left[ \log p(x_i, z) \right] \right) + \mathbb{H}\left( q(z) \right)
\end{align*}
$$

and $\mathbb{H} \left(q(z) \right)$ denotes the [(differential) entropy](https://www.wikiwand.com/en/Differential_entropy) of $q(z)$.

Assuming joint $p(x_i, z)$ and the entropy $\mathbb{H}\left(q(z)\right)$ are both tractable, we can use a Monte-Carlo for the remaining expectation. This leaves us with the following tractable expression

$$
\underset{q}{\mathrm{argmin}} \ \mathrm{D_{KL}} \left( q(z), p(z \mid \{ x_i \}_{i = 1}^n) \right) \approx \underset{q}{\mathrm{argmax}} \ \widehat{\mathrm{ELBO}}(q)
$$

where

$$
\widehat{\mathrm{ELBO}}(q) = \frac{1}{m} \left( \sum_{k = 1}^m \sum_{i = 1}^n \log p(x_i, z_k) \right) + \mathbb{H} \left(q(z)\right) \quad \text{where} \quad z_k \sim q(z) \quad \forall k = 1, \dots, m.
$$

Hence, as long as we can sample from $q(z)$ somewhat efficiently, we can indeed minimize the KL-divergence! Neat, eh?

Sidenote: in the case where $q(z)$ is tractable but $\mathbb{H} \left(q(z) \right)$ is _not_ , we can use an Monte-Carlo estimate for this term too but this generally results in a higher-variance estimate.

Also, I fooled you real good: the ELBO _isn't_ an arbitrary name, hah! In fact it's an abbreviation for the **expected lower bound (ELBO)** because it, uhmm, well, it's the _expected_ lower bound (remember $\mathrm{D_{KL}} \ge 0$). Yup.

## Maximizing the ELBO

Finding the optimal $q$ over _all_ possible densities of course isn't feasible. Instead we consider a family of _parameterized_ densities $\mathscr{D}\_{\Theta}$ where $\Theta$ denotes the space of possible parameters. Each density in this family $q_{\theta} \in \mathscr{D}\_{\Theta}$ is parameterized by a unique $\theta \in \Theta$. Moreover, we'll assume
1. $q\_{\theta}(z)$, i.e. evaluating the probability density $q$ at any point $z$, is differentiable
2. $z \sim q_{\theta}(z)$, i.e. the process of sampling from $q_{\theta}(z)$, is differentiable

(1) is fairly straight-forward, but (2) is a bit tricky. What does it even mean for a _sampling process_ to be differentiable? This is quite an interesting problem in its own right and would require something like a [50-page paper to properly review the different approaches (highly recommended read)](https://arxiv.org/abs/1906.10652).

We're going to make use of a particular such approach which goes under a bunch of different names: _reparametrization trick_, _path derivative_, etc. This refers to making the assumption that all elements $q\_{\theta} \in \mathscr{Q}\_{\Theta}$ can be considered as reparameterizations of some base density, say $\bar{q}(z)$. That is, if $q\_{\theta} \in \mathscr{Q}\_{\Theta}$ then

$$
z \sim q_{\theta}(z) \quad \iff \quad z := g_{\theta}(\tilde{z}) \quad \text{where} \quad \bar{z} \sim \bar{q}(z)
$$

for some function $g\_{\theta}$ differentiable wrt. $\theta$. So all $q\_{\theta} \in \mathscr{Q}\_{\Theta}$ are using the *same* reparameterization-function $g$ but each $q\_{\theta}$ correspond to different choices of $\theta$ for $f\_{\theta}$.

Under this assumption we can differentiate the sampling process by taking the derivative of $g_{\theta}$ wrt. $\theta$, and thus we can differentiate the entire $\widehat{\mathrm{ELBO}}(q_{\theta})$ wrt. $\theta$! With the gradient available we can either try to solve for optimality either by setting the gradient equal to zero or maximize $\widehat{\mathrm{ELBO}}(q_{\theta})$ stepwise by traversing $\mathscr{Q}_{\Theta}$ in the direction of steepest ascent. For the sake of generality, we're going to go with the stepwise approach.

With all this nailed down, we eventually reach the section on **Automatic Differentiation Variational Inference (ADVI)**.

## Automatic Differentiation Variational Inference (ADVI)

So let's revisit the assumptions we've made at this point:
1. The variational posterior $q\_{\theta}$ is in a parameterized family of densities denoted $\mathscr{Q}\_{\Theta}$, with $\theta \in \Theta$.
2. $\mathscr{Q}\_{\Theta}$ is a space of _reparameterizable_ densities with $\bar{q}(z)$ as the base-density.
3. The parameterization function $g\_{\theta}$ is differentiable wrt. $\theta$.
4. Evaluation of the probability density $q\_{\theta}(z)$ is differentiable wrt. $\theta$.
5. $\mathbb{H}\left(q_{\theta}(z)\right)$ is tractable.
6. Evaluation of the joint density $p(x, z)$ is tractable and differentiable wrt. $z$
7. The support of $p(z \mid x)$ is a subspace of the support of $q(z)$: $\mathrm{supp}\left(p(z \mid x)\right) \subseteq \mathrm{supp}\left(q(z)\right)$.

All of these are not *necessary* to do VI, but they are very convenient and results in a fairly flexible approach. One distribution which has a density satisfying all of the above assumptions _except_ (7) (we'll get back to this in second) for any tractable and differentiable $p(z \mid \{ x_i \}_{i = 1}^n)$ is the good ole' Gaussian/normal distribution:

$$
z \sim \mathcal{N}(\mu, \Sigma) \quad \iff \quad z = g_{\mu, L}(\bar{z}) := \mu + L^T \tilde{z} \quad \text{where} \quad \bar{z} \sim \bar{q}(z) := \mathcal{N}(1_d, I_{d \times d})
$$

where $\Sigma = L L^T$, with $L$ obtained from the Cholesky-decomposition. Abusing notation a bit, we're going to write

$$
\theta = (\mu, \Sigma) := (\mu_1, \dots, \mu_d, L_{11}, \dots, L_{1, d}, L_{2, 1}, \dots, L_{2, d}, \dots, L_{d, 1}, \dots, L_{d, d}).
$$

With this assumption we finally have a tractable expression for $\widehat{\mathrm{ELBO}}(q\_{\mu, \Sigma})$! Well, assuming (7) is holds. Since a Gaussian has non-zero probability on the entirety of $\mathbb{R}^d$, we also require $p(z \mid \\{ x_i \\}\_{i = 1}^n)$ to have non-zero probability on all of $\mathbb{R}^d$.

Though not necessary, we'll often make a *mean-field* assumption for the variational posterior $q(z)$, i.e. assume independence between the latent variables. In this case, we'll write

$$
\theta = (\mu, \sigma^2) := (\mu_1, \dots, \mu_d, \sigma_1^2, \dots, \sigma_d^2).
$$

### Examples
As a (trivial) example we could apply the approach described above to is the following generative model for $p(z \mid \\{ x_i \\}\_{i = 1}^n)$:

$$
\begin{align*}
    m &\sim \mathcal{N}(0, 1) \\\\
    x_i &\overset{\text{i.i.d.}}{=} \mathcal{N}(m, 1), \quad i = 1, \dots, n.
\end{align*}
$$

In this case $z = m$ and we have the posterior defined $p(m \mid \\{ x\_i \\}\_{i = 1}^n) = p(m) \prod\_{i = 1}^n p(x\_i \mid m)$. Then the variational posterior would be

$$
q_{\mu, \sigma} = \mathcal{N}(\mu, \sigma^2), \quad \text{where} \quad \mu \in \mathbb{R}, \ \sigma^2 \in \mathbb{R}^{ + }.
$$

And since prior of $m$, $\mathcal{N}(0, 1)$, has non-zero probability on the entirety of $\mathbb{R}$, same as $q(m)$, i.e. assumption (7) above holds, everything is fine and life is good.

But what about this generative model for $p(z \mid \{ x_i \}_{i = 1}^n)$:

$$
\begin{align*}
    s &\sim \mathrm{InverseGamma}(2, 3), \\\\
    m &\sim \mathcal{N}(0, s), \\\\
    x_i &\overset{\text{i.i.d.}}{=} \mathcal{N}(m, s), \quad i = 1, \dots, n,
\end{align*}
$$

with posterior $p(s, m \mid \\{ x\_i \\}\_{i = 1}^n) = p(s) p(m \mid s) \prod\_{i = 1}^n p(x_i \mid s, m)$ and the mean-field variational posterior $q(s, m)$ will be

$$
q_{\mu_1, \mu_2, \sigma_1^2, \sigma_2^2}(s, m) = p_{\mathcal{N}(\mu_1, \sigma_1^2)}(s)\ p_{\mathcal{N}(\mu_2, \sigma_2^2)}(m),
$$

where we've denoted the evaluation of the probability density of a Gaussian as $p_{\mathcal{N}(\mu, \sigma^2)}(x)$.

Observe that $\mathrm{InverseGamma}(2, 3)$ has non-zero probability only on $\mathbb{R}^{ + } := (0, \infty)$ which is clearly not all of $\mathbb{R}$ like $q(s, m)$ has, i.e.

$$
\mathrm{supp} \left( q(s, m) \right) \not\subseteq \mathrm{supp} \left( p(z \mid \{ x_i \}_{i = 1}^n) \right).
$$

Recall from the definition of the KL-divergence that when this is the case, the KL-divergence isn't well defined. This gets us to the *automatic* part of ADVI.

### "Automatic"? How?

For a lot of the standard (continuous) densities $p$ we can actually construct a probability density $\tilde{p}$ with non-zero probability on all of $\mathbb{R}$ by *transforming* the "constrained" probability density $p$ to $\tilde{p}$. In fact, in these cases this is a one-to-one relationship. As we'll see, this helps solve the support-issue we've been going on and on about.

#### Transforming densities using change of variables

If we want to compute the probability of $x$ taking a value in some set $A \subseteq \mathrm{supp} \left( p(x) \right)$, we have to integrate $p(x)$ over $A$, i.e.

$$
\mathbb{P}_p(x \in A) = \int_A p(x) \mathrm{d}x.
$$

This means that if we have a differentiable bijection $f: \mathrm{supp} \left( q(x) \right) \to \mathbb{R}^d$ with differentiable inverse $f^{-1}: \mathbb{R}^d \to \mathrm{supp} \left( p(x) \right)$, we can perform a change of variables

$$
\mathbb{P}_p(x \in A) = \int_{f^{-1}(A)} p \left(f^{-1}(y) \right) \ \left| \det \mathcal{J}_{f^{-1}}(y) \right| \ \mathrm{d}y,
$$

where $\mathcal{J}\_{f^{-1}}(x)$ denotes the jacobian of $f^{-1}$ evaluted at $x$. Observe that this defines a probability distribution

$$
\mathbb{P}_{\tilde{p}}\left(y \in f^{-1}(A) \right) = \int_{f^{-1}(A)} \tilde{p}(y) \mathrm{d}y,
$$

since $f^{-1}\left(\mathrm{supp} (p(x)) \right) = \mathbb{R}^d$ which has probability 1. This probability distribution has *density* $\tilde{p}(y)$ with $\mathrm{supp} \left( \tilde{p}(y) \right) = \mathbb{R}^d$, defined

$$
\tilde{p}(y) = p \left( f^{-1}(y) \right) \ \left| \det \mathcal{J}_{f^{-1}}(y) \right|
$$

or equivalently

$$
\tilde{p} \left( f(x) \right) = \frac{p(x)}{\big| \det \mathcal{J}_{f}(x) \big|}
$$

due to the fact that

$$
\big| \det \mathcal{J}_{f^{-1}}(y) \big| = \big| \det \mathcal{J}_{f}(x) \big|^{-1}
$$

*Note: it's also necessary that the log-abs-det-jacobian term is non-vanishing. This can for example be accomplished by assuming $f$ to also be elementwise monotonic.*

#### Back to VI

So why is this is useful? Well, we're looking to generalize our approach using a normal distribution to cases where the supports don't match up. How about defining $q(z)$ by

$$
\begin{align*}
  \eta &\sim \mathcal{N}(\mu, \Sigma), \\\\
  z &= f^{-1}(\eta),
\end{align*}
$$

where $f^{-1}: \mathbb{R}^d \to \mathrm{supp} \left( p(z \mid x) \right)$ is a differentiable bijection with differentiable inverse. Then $z \sim q\_{\mu, \Sigma}(z) \implies z \in \mathrm{supp} \left( p(z \mid x) \right)$ as we wanted. The resulting variational density is

$$
q_{\mu, \Sigma}(z) = p_{\mathcal{N}(\mu, \Sigma)}\left( f(z) \right) \ \big| \det \mathcal{J}_{f}(z) \big|.
$$

Note that the way we've constructed $q(z)$ here is basically a reverse of the approach we described above. Here we sample from a distribution with support on $\mathbb{R}$ and transform *to* $\mathrm{supp} \left( p(z \mid x) \right)$.

If we want to write the ELBO explicitly in terms of $\eta$ rather than $z$, the first term in the ELBO becomes

$$
\begin{align*}
  \mathbb{E}_{z \sim q_{\mu, \Sigma}(z)} \left[ \log p(x_i, z) \right] &= \mathbb{E}_{\eta \sim \mathcal{N}(\mu, \Sigma)} \Bigg[ \log \frac{p\left(x_i, f^{-1}(\eta) \right)}{\big| \det \mathcal{J}_{f^{-1}}(\eta) \big|} \Bigg] \\\\
  &= \mathbb{E}_{\eta \sim \mathcal{N}(\mu, \Sigma)} \left[ \log p\left(x_i, f^{-1}(\eta) \right) \right] - \mathbb{E}_{\eta \sim \mathcal{N}(\mu, \Sigma)} \left[ \left| \det \mathcal{J}_{f^{-1}}(\eta) \right| \right].
\end{align*}
$$

The entropy is invariant under change of variables, thus $\mathbb{H} \left(q\_{\mu, \Sigma}(z)\right)$ is simply the entropy of the normal distribution which is known analytically. 

Hence, the resulting empirical estimate of the ELBO is

$$
\begin{align*}
\widehat{\mathrm{ELBO}}(q_{\mu, \Sigma}) &= \frac{1}{m} \bigg( \sum_{k = 1}^m \sum_{i = 1}^n \left(\log p\left(x_i, f^{-1}(\eta_k)\right) - \log \big| \det \mathcal{J}_{f^{-1}}(\eta_k) \big| \right) \bigg) + \mathbb{H} \left(p_{\mathcal{N}(\mu, \Sigma)}(z)\right) \\\\
& \text{where} \quad z_k  \sim \mathcal{N}(\mu, \Sigma) \quad \forall k = 1, \dots, m
\end{align*}.
$$

And maximizing this wrt. $\mu$ and $\Sigma$ is what's referred to as **Automatic Differentiation Variational Inference (ADVI)**!

Now if you want to try it out, [check out the tutorial on how to use ADVI in Turing.jl](../../tutorials/09-variational-inference)!

