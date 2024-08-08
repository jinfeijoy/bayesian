# Bayesian

![image](https://github.com/user-attachments/assets/cbd5fd2f-69b6-472b-9fa7-7e19c0df4c57)


## Stage 1: [coursera specialization](https://www.coursera.org/specializations/bayesian-statistics) (July 22 - Aug 16) 
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
       ![image](https://github.com/user-attachments/assets/7f2b8bfd-8859-4a81-a668-47c2e7b289c5)
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
  * code: [anova](https://d3c33hcgiwev3.cloudfront.net/_79e4d8c2bf9cf1a0589ba96902cf8fee_lesson_08.html?Expires=1722384000&Signature=jHJZ9HmdYtXuB5vj25DXtlYRoWNZdkKQSlQLT1C7rEv-TrNIHc8s4Zq~4ky88PIzUaUko5PAbrtfwlXZu0OKh0wmABioEKNrWAe3v~0TQpFM1u-DSMaheyKiz42GgzKCr7JQLJsOV3aiGiJDWg9--4BFOmt7ydpiSq5FlVPNQpw_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A),[linear](https://d3c33hcgiwev3.cloudfront.net/_88ead38436bb1be19e37ed542ac9b49a_lesson_07.html?Expires=1722384000&Signature=TfIgZFyxqD8ZiG1NYF7K9G6Wrm9S-xAuviQdTQ06iFpC8jeHxegWZONJ71UdSoo~ZA4aqXWnGjbfWwAe1llNqudx3bIKObfZafWp6ryuJBYX5DF7dNcGOO~xMAx9rJkXExM9CDLFXtNos1uzh2nDpbBXobE2B1XYraiYsElDQ1U_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A),[logistic](https://d3c33hcgiwev3.cloudfront.net/_788f56b413be4c89c5a13d4b1faa2891_lesson_09.html?Expires=1722384000&Signature=TtRuCUGxJr6ApfOE5Ug6abtA0ya3TqfCKeYnzyiwJDw9KAw9fkeQe9O98aHZItvF5sxOVJxdNc5VEmA6pGemc~~qROkQIRLLZxgtt1-8zfjYu7HUpwLpWYTFDNe0RiZk9JSNpO6BlBTrf7DNj15iON6P7LEKwVGu9wdXzIgu~S0_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
          [multi factor anova](https://d3c33hcgiwev3.cloudfront.net/_934923025d1686bb6a850a5858fc97d7_lesson_08multipleANOVA.html?Expires=1722384000&Signature=OUEobQJVfu~KHoHFhvDC7HJNb5chnUC1aTP0JgZ8TwphO6oHJkqR1ZJlWGNZ5vqbN8D-FveemPBThWqf4SvpSzs9voGL5HeFAGuB9LuHS5i7OeXrZI5biCZqBIeKXtqQui0IY6AElzBmQYvtnLZt1cfJSypdSGk7E3-hU7dgwwM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
  * poison distribution, if model doesn't fit very well, can try negative binomial distribution
     * [code](https://d3c33hcgiwev3.cloudfront.net/_2e3249041d23214d57cf3a2e01a69d5f_lesson_10.html?Expires=1722470400&Signature=BSw7zzNwj0rCE02-93LL1kWaXNgjF2yJ2NPjQBMUjPa7pwh6gGIwllW0NiLCtamOUk7g9nTINY548DSkyZnLF3kKMN23ggXEkOtZClbbrsVFrMAC0WuvHjhlPq4b64Frc0zjPCyqM-IpwS9s5Rm-QovRg8qRGr0EKjCvyNNZofg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
  * hierarchical models：Being able to account for relationships in the data while estimating everything with the single model is a primary advantage of using hierarchical models.
     * [code](https://d3c33hcgiwev3.cloudfront.net/_c1ff186bc258dd9175da24316a7dcc25_lesson_11.html?Expires=1722470400&Signature=apP3Qa80MYOYmIuAZIFdiDt2RwXc-147QP1ywW9RX9bUhSzlLGFXEm5Nih9vf3KD~u57~WZV0pdpIIw5k4AAGxa2xqhoL2CesYiha36Iwt0LDY40YH50FVXgXUC9lRzp8X2-Jsh0ZnORP8KTYbcle7iZey5O6r0Iioe32mFGfhY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

* Course 3: (July 31.2/Aug 2.1/6.1/7.1)
  * mixture hierarchical: So for each observation that you want to generate, you're going to do two steps. 
    * In the first step, you're going to randomly sample the component indicators. So you are going to sample c_i with probability given by the weights Omega_1 up to Omega_K. 
    * Given the value of c_i that you just sampled, you are going to generate X_i from g of c_i. So this just means that sample X_i from the c_i component of the mixture. Again, I need to emphasize that this is a very simple algorithm that allows you to generate the random values. 
  * MCMC for bayesian inference:
    * sample code: [1](https://www.coursera.org/learn/mixture-models/supplement/HgGgw/sample-code-for-mcmc-example-1), [2](https://www.coursera.org/learn/mixture-models/supplement/wqh0u/sample-code-for-mcmc-example-2)
    *  ![image](https://github.com/user-attachments/assets/63a954b1-8f33-4838-90d2-1fb2cacd92c6)
    *  ![image](https://github.com/user-attachments/assets/619e1c2c-4c8d-450a-9f1c-84da9860cc40)
    *  ![image](https://github.com/user-attachments/assets/3d46d140-aa81-4242-9f2e-e703134e70ee)
    *  ![image](https://github.com/user-attachments/assets/63138a55-ff75-40ca-b4d8-7fe2d747ca64)
    *  ![image](https://github.com/user-attachments/assets/ebe25f9e-c853-4263-9da7-5308d0c9e96e)
  * Application 
    * density estimation (kernal density estimation / mixture model)
    * clustering (k means/mixture model with EM)
    * classification 
  * BIC: bayesian information criteria, model accuracy + model complexity, lower better

* Course 4: (Aug 8.1/12.1/13.1/14.1)
  * bayesian time series: given prior distribution of parameter, get function of model, get posterior function of parameter with conditional term, update posterior distribution with function and input data
  * AR(p): linear model with log term
  * normal dynamic linear model
