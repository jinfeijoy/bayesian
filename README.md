# Bayesian

## Stage 1: [coursera specialization](https://www.coursera.org/specializations/bayesian-statistics) (July 22 - Aug 9) 
* Course 1: (July 22/23)
  * probability:
    * classical - equally likely
    * frequentist - relative frequency
    * bayesian - personal perspective  
  * different distributions 
    * given distribution, initial prior distribution, update distribution parameters (mean/std) with given record, then get posterior distribution.
* Course 2: (July 24/25/26/29/30)
  * statistical model:
     * quantify uncertainty / inference / measure support for hypotheses / prediction 
  * bayesian model:
     * different layers: ![image](https://github.com/user-attachments/assets/c9f26147-f894-45c1-b0f6-06ddc43edc25)
     * [common probability distribution](https://d3c33hcgiwev3.cloudfront.net/_d7c17d00198049b1ccfdf72d2831d2be_Distributions.pdf?Expires=1721952000&Signature=RQzRCiRolMCjHfzo8GPjuXmigL2eFlN04lpij7VErwUSqKInvC-95BMfx4ptMw00HPYK21KkZlD0lw8AyHbJuUSZ-JWYEyimnYeCxQkNNsI6nn5XR0sUqGM~dukoLY6DIpDWEnm2r1se~5PqsPnLCRVKxPuxH~IDPoMO97fDfMk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A) 
  * mcmc: 
     * [algorithm](https://d3c33hcgiwev3.cloudfront.net/_caf094bf3db01507bea6305d040883e4_lesson_04.html?Expires=1722124800&Signature=bX~KLfjoymB0cNNuz1YKmj97Vm0yXwRqTUslMNiFF9RlZjy5RMiT-fSRIfmIP4iKtGuwgIAwQXRm2cTv~NfQN85MFDUQQvsjBX-fGfoXsyWmc9q5jKiloe9Ml5l9BY73-AjlOLNNOHQ7cio0lNXhXyXuI5CmX9Fn9OHqmEtE41s_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
     * ideal accept rate for random walk is between 0.23-0.5, to increase accept rate we need to decrease standard deviation, to get a stationary distribution
     * package: jags (pyjags in python)
     * gibsampling: can be used to generate sampling with more than 1 parameter. first to identify distribution, then to get the parameter update function for posterier disyribution, then update parameters with mcmc 
     * once generate samples, can calculate autocorrelation to check with how many lags the autocorrelation become 0. then can only take values every n lags to avoid autocorrelations to increase randomize
     * multiple chain to check convergence, can use Gelman-rubin diagnostics, if value close to 1 then means converge, otherwise is not converge
  * jags:
     * define model string (include data y, distribution function, prior distribution for parameters, 
     * specify the data, provide initial value for parameters
     * use jags to get multiple chains
     * combine chains and save samples (posterior distribution) 
     * test convergence (trace plot, gelman diag, auto correlation, effective sample size)
     * check residuals 
     * average posterior distribution to get parameter value
     * get y based on updated parameters, then check residual to see if its random distributed, then check qq plot to see if there are outliers, find out outliers based on residuals
     * if model is not as good as expected, thete are two ways to pick another model: 
        * adding more variables for existing distribution mu 
        * change existing distribution to another distribution
     * evaluation matrics: DIC (Deviance information criteria)
  * anova
     * used for category variable to check within group and out of group difference


* Course 3: (July 31/Aug 1/2/5)
* Course 4: (Aug 6/7/8/9)
