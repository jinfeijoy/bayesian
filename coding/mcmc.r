
# install.packages('rjags')
library("rjags")

############################################ MCMC ######################################################
# https://d3c33hcgiwev3.cloudfront.net/_4ec003e4706226af504d524aafb2c527_JAGSintro.html?Expires=1723161600&Signature=kNy14bJ89J3JeUfhYBDxgaHwtOF5LY9DK1j70DhkF0pp-k5ZGgBXh7E7NEh8TaPXPAHVu4fl0ZVMmT9k6fxZbwovxByeJnJoEMZuEEATcqJ2WEkEDkW9d5hxBUj1nZPXFSVdaX1ypV2tevRTrysVo~nPSAhOYXINcQRjw4V3DkE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A

# 1. specify the model
mod_string = " model {
  for (i in 1:n) {
    y[i] ~ dnorm(mu, 1.0/sig2)
  }
  mu ~ dt(0.0, 1.0/1.0, 1.0) # location, inverse scale, degrees of freedom
  sig2 = 1.0
} "

# 2. setup the model
set.seed(50)
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
n = length(y)

data_jags = list(y=y, n=n)
params = c("mu")

inits = function() {
  inits = list("mu"=0.0)
} # optional (and fixed)

mod = jags.model(textConnection(mod_string), data=data_jags, inits=inits)

# 3. run MCMC sampler
update(mod, 500) # burn-in

mod_sim = coda.samples(model=mod,
                       variable.names=params,
                       n.iter=1000)

# 4. post processing
summary(mod_sim)

library("coda")
plot(mod_sim)

traceplot(as.mcmc(mod_sim))

############################################ Metropolis-Hastings ############################################
# https://d3c33hcgiwev3.cloudfront.net/_caf094bf3db01507bea6305d040883e4_lesson_04.html?Expires=1723161600&Signature=eICf1YVPZkxGmPvL0g51R-tlHNS5u9KQjb2quYNBNGls3UhvCi8wB6Lyh7DTtiL7tbTeOXaJCwBJzTKb9kRClizMbRYzNuy1i8imiYPeeaVa4yCuGzoI4-sklKCh2T9sAD8cw9G5otxvZ2F-ToAkGg64FeVwEDdJx9lrvL1HbH8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A
lg = function(mu, n, ybar) {
  mu2 = mu^2
  n * (ybar * mu - mu2 / 2.0) - log(1 + mu2)
}
mh = function(n, ybar, n_iter, mu_init, cand_sd) {
  ## Random-Walk Metropolis-Hastings algorithm
  
  ## step 1, initialize
  mu_out = numeric(n_iter)
  accpt = 0
  mu_now = mu_init
  lg_now = lg(mu=mu_now, n=n, ybar=ybar)
  
  ## step 2, iterate
  for (i in 1:n_iter) {
    ## step 2a
    mu_cand = rnorm(n=1, mean=mu_now, sd=cand_sd) # draw a candidate
    
    ## step 2b
    lg_cand = lg(mu=mu_cand, n=n, ybar=ybar) # evaluate log of g with the candidate
    lalpha = lg_cand - lg_now # log of acceptance ratio
    alpha = exp(lalpha)
    
    ## step 2c
    u = runif(1) # draw a uniform variable which will be less than alpha with probability min(1, alpha)
    if (u < alpha) { # then accept the candidate
      mu_now = mu_cand
      accpt = accpt + 1 # to keep track of acceptance
      lg_now = lg_cand
    }
    
    ## collect results
    mu_out[i] = mu_now # save this iteration's value of mu
  }
  
  ## return a list of output
  list(mu=mu_out, accpt=accpt/n_iter)
}
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
ybar = mean(y)
n = length(y)
hist(y, freq=FALSE, xlim=c(-1.0, 3.0)) # histogram of the data
curve(dt(x=x, df=1), lty=2, add=TRUE) # prior for mu
points(y, rep(0,n), pch=1) # individual data points
points(ybar, 0, pch=19) # sample mean

set.seed(43) # set the random seed for reproducibility
post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=3.0)
str(post)

traceplot(as.mcmc(post$mu))

post$mu_keep = post$mu[-c(1:100)] # discard the first 200 samples
plot(density(post$mu_keep, adjust=2.0), main="", xlim=c(-1.0, 3.0), xlab=expression(mu)) # plot density estimate of the posterior
curve(dt(x=x, df=1), lty=2, add=TRUE) # prior for mu
points(ybar, 0, pch=19) # sample mean

curve(0.017*exp(lg(mu=x, n=n, ybar=ybar)), from=-1.0, to=3.0, add=TRUE, col="blue") # approximation to the true posterior in 

############################################ Coverage ############################################
set.seed(61)
post0 = mh(n=n, ybar=ybar, n_iter=10e3, mu_init=0.0, cand_sd=0.9)
traceplot(as.mcmc(post0$mu[-c(1:500)]))

autocorr.plot(as.mcmc(post0$mu))
autocorr.diag(as.mcmc(post0$mu))
coda::effectiveSize(as.mcmc(post0$mu))
str(post0)

set.seed(61)
post2 = mh(n=n, ybar=ybar, n_iter=100e3, mu_init=0.0, cand_sd=0.04)
coda::traceplot(as.mcmc(post2$mu))
coda::effectiveSize(as.mcmc(post2$mu))
coda::autocorr.plot(as.mcmc(post2$mu), lag.max=500)
str(post2)

raftery.diag(as.mcmc(post0$mu))

############################################ Multiple chains, Gelman-Rubin ############################################
set.seed(61)

nsim = 500
post1 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=15.0, cand_sd=0.4)
post1$accpt
post2 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=-5.0, cand_sd=0.4)
post2$accpt
post3 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=7.0, cand_sd=0.1)
post3$accpt
post4 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=23.0, cand_sd=0.5)
post4$accpt
post5 = mh(n=n, ybar=ybar, n_iter=nsim, mu_init=-17.0, cand_sd=0.4)
post5$accpt
pmc = mcmc.list(as.mcmc(post1$mu), as.mcmc(post2$mu), 
                as.mcmc(post3$mu), as.mcmc(post4$mu), as.mcmc(post5$mu))
str(pmc)

traceplot(pmc)
gelman.diag(pmc)
gelman.plot(pmc)

############################################ Monte Carlo estimation ############################################
nburn = 1000 # remember to discard early iterations
post0$mu_keep = post0$mu[-c(1:1000)]
summary(as.mcmc(post0$mu_keep))
mean(post0$mu_keep > 1.0) # posterior probability that mu  > 1.0
