

Folder layout:

* figures/
	* data/
		* plot of data for each condition
		* data used for model fit is in log2_st0/
	* standard/
		* standard growth results, png's here are model fit to data
		* od_delta_deriv_mu.png
			* mean difference between ura3 and mutant strain growth rate
		* od_delta_deriv_prob.png
			* probability of difference between ura3 and mutant strain growth rate
		* od_delta_mu.png
			* mean difference between ura3 and mutant strain growth
		* od_delta_prob.png
			* probability of difference between ura3 and mutant strain growth
		* od_delta/
			* OD delta of growth level for each strain
		* od_delta_deriv/
			* OD delta of the derivative (growth rate)
	* paraquat/
		* same as standard/
	* osmotic/
		* same as standard/
	* heatshock/
			* same as standard/

# Methods

## Data preparation

Data was $log_2$ transformed and each condition was normalized so the average of
the first time point for each condition was equal to zero.

## Analysis

### Model

Each combination of $\Delta ura3$ and mutant strain data under each condition
was modeled with a Gaussian process (GP):

\[y \sim GP(\mu(x),K(x_1,x_2)).\]

As is standard, $\mu(x)$ was assumed to be 0 for all $x$. The kernel function
used was the radial basis function (RBF):

\[K_{\text{RBF}}(x_1,x_2) = \sigma^2 \cdot exp\Big(\sum_{i=1}^p \frac{-||x_{1,i}-x_{2,i}||^2}{\ell^2_i}\Big).\]

For standard conditions and heatshock, growth was compared between $\Delta ura3$
mutant strain under the condition of interest. In this case, time and genetic background are the
covariates of the model, e.g. $x = \{\text{time, strain}\}$, where $\text{strain}\in\{0,1\}$
is equal to 1 for the mutant strain, 0 otherwise. For osmotic and paraquat stress,
the covariates were expanded to include a stress indicator, as well as an interaction
between mutant strain and stress condition. E.g. $x = \{\text{time, strain, stress, strain}\times\text{stress}\}$.
Where $\text{stress} \in \{0,1\}$ is 1 for stress conditions, 0 otherwise, and
$\text{strain}\times\text{stress} \in \{0,1\}$ is equal to 1 for mutant strain
under stress conditions, 0 otherwise.
