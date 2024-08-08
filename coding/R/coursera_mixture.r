############################################ mixture ############################################
library(fourPNO)
#### example of bimodal mixture of gaussians
## Mixture of univariate Gaussians, bimodal
x = seq(-5, 12, length=100)
y = 0.6*dnorm(x, 0, 1) + 0.4*dnorm(x, 5, 2)
par(mar=c(4,4,1,1)+0.1)
plot(x, y, type="l", ylab="Density", las=1, lwd=2)

#### Example of a unimodal and skewed mixture of Gaussians
# Zero inflated negative binomial distribution
x = seq(0, 15)
y = dnbinom(x, 8, 0.6)
z = 0.2*c(1,rep(0,length(x)-1)) + (1-0.2)*y
par(mfrow=c(2,1))
par(mar=c(4,4,2,2)+0.1)
barplot(y, names.arg=x, las=1, xlab = "x", ylab="Probability", 
        border=NA, main="Negative Binomial")
par(mar=c(4,4,1,1)+0.1)
barplot(z, names.arg=x, las=1, xlab = "x", ylab="Probability", 
        border=NA, main="Zero-inflated Negative Binomial")


#### Sample code for simulating from a Mixture Model
# Generate n observations from a mixture of two Gaussian 
# distributions
n     = 50           # Size of the sample to be generated
w     = c(0.6, 0.4)  # Weights
mu    = c(0, 5)      # Means
sigma = c(1, 2)      # Standard deviations
cc    = sample(1:2, n, replace=T, prob=w)
x     = rnorm(n, mu[cc], sigma[cc])
    
# Plot f(x) along with the observations 
# just sampled
xx = seq(-5, 12, length=200)
yy = w[1]*dnorm(xx, mu[1], sigma[1]) + 
     w[2]*dnorm(xx, mu[2], sigma[2])
par(mar=c(4,4,1,1)+0.1)
plot(xx, yy, type="l", ylab="Density", xlab="x", las=1, lwd=2)
points(x, y=rep(0,n), pch=1)

####

####

####

####

####

####


############################################ EM ############################################

#### Example of an EM algorithm for fitting a location mixture of 2 Gaussian components
#### The algorithm is tested using simulated data

## Clear the environment and load required libraries
rm(list=ls())
set.seed(81196)    # So that results are reproducible (same simulated data every time)

## Generate data from a mixture with 2 components
KK         = 2
w.true     = 0.6  # True weights associated with the components
mu.true    = rep(0, KK)
mu.true[1] = 0   # True mean for the first component
mu.true[2] = 5   # True mean for the second component
sigma.true = 1   # True standard deviation of all components
n  = 120         # Number of observations to be generated
cc = sample(1:KK, n, replace=T, prob=c(w.true,1-w.true))
x  = rep(0, n)
for(i in 1:n){
  x[i] = rnorm(1, mu.true[cc[i]], sigma.true)
}

# Plot the true density
par(mfrow=c(1,1))
xx.true = seq(-8,11,length=200)
yy.true = w.true*dnorm(xx.true, mu.true[1], sigma.true) + 
          (1-w.true)*dnorm(xx.true, mu.true[2], sigma.true) 
plot(xx.true, yy.true, type="l", xlab="x", ylab="True density", lwd=2)
points(x, rep(0,n), col=cc)

## Run the actual EM algorithm
## Initialize the parameters
w     = 1/2                         #Assign equal weight to each component to start with
mu    = rnorm(KK, mean(x), sd(x))   #Random cluster centers randomly spread over the support of the data
sigma = sd(x)                       #Initial standard deviation

# Plot the initial guess for the density
xx = seq(-8,11,length=200)
yy = w*dnorm(xx, mu[1], sigma) + (1-w)*dnorm(xx, mu[2], sigma)
plot(xx, yy, type="l", ylim=c(0, max(yy)), xlab="x", ylab="Initial density")
points(x, rep(0,n), col=cc)

s  = 0
sw = FALSE
QQ = -Inf
QQ.out = NULL
epsilon = 10^(-5)


##Checking convergence of the algorithm
while(!sw){
  ## E step
  v = array(0, dim=c(n,KK))
  v[,1] = log(w) + dnorm(x, mu[1], sigma, log=TRUE)    #Compute the log of the weights
  v[,2] = log(1-w) + dnorm(x, mu[2], sigma, log=TRUE)  #Compute the log of the weights
  for(i in 1:n){
    v[i,] = exp(v[i,] - max(v[i,]))/sum(exp(v[i,] - max(v[i,])))  #Go from logs to actual weights in a numerically stable manner
  }
  
  ## M step
  # Weights
  w = mean(v[,1])
  mu = rep(0, KK)
  for(k in 1:KK){
    for(i in 1:n){
      mu[k]    = mu[k] + v[i,k]*x[i]
    }
    mu[k] = mu[k]/sum(v[,k])
  }
  # Standard deviations
  sigma = 0
  for(i in 1:n){
    for(k in 1:KK){
      sigma = sigma + v[i,k]*(x[i] - mu[k])^2
    }
  }
  sigma = sqrt(sigma/sum(v))
  
  ##Check convergence
  QQn = 0
  for(i in 1:n){
    QQn = QQn + v[i,1]*(log(w) + dnorm(x[i], mu[1], sigma, log=TRUE)) +
                v[i,2]*(log(1-w) + dnorm(x[i], mu[2], sigma, log=TRUE))
  }
  if(abs(QQn-QQ)/abs(QQn)<epsilon){
    sw=TRUE
  }
  QQ = QQn
  QQ.out = c(QQ.out, QQ)
  s = s + 1
  print(paste(s, QQn))
  
  #Plot current estimate over data
  layout(matrix(c(1,2),2,1), widths=c(1,1), heights=c(1.3,3))
  par(mar=c(3.1,4.1,0.5,0.5))
  plot(QQ.out[1:s],type="l", xlim=c(1,max(10,s)), las=1, ylab="Q", lwd=2)
  
  par(mar=c(5,4,1.5,0.5))
  xx = seq(-8,11,length=200)
  yy = w*dnorm(xx, mu[1], sigma) + (1-w)*dnorm(xx, mu[2], sigma)
  plot(xx, yy, type="l", ylim=c(0, max(c(yy,yy.true))), main=paste("s =",s,"   Q =", round(QQ.out[s],4)), lwd=2, col="red", lty=2, xlab="x", ylab="Density")
  lines(xx.true, yy.true, lwd=2)
  points(x, rep(0,n), col=cc)
  legend(6,0.22,c("Truth","Estimate"),col=c("black","red"), lty=c(1,2))
}


#Plot final estimate over data
layout(matrix(c(1,2),2,1), widths=c(1,1), heights=c(1.3,3))
par(mar=c(3.1,4.1,0.5,0.5))
plot(QQ.out[1:s],type="l", xlim=c(1,max(10,s)), las=1, ylab="Q", lwd=2)

par(mar=c(5,4,1.5,0.5))
xx = seq(-8,11,length=200)
yy = w*dnorm(xx, mu[1], sigma) + (1-w)*dnorm(xx, mu[2], sigma)
plot(xx, yy, type="l", ylim=c(0, max(c(yy,yy.true))), main=paste("s =",s,"   Q =", round(QQ.out[s],4)), lwd=2, col="red", lty=2, xlab="x", ylab="Density")
lines(xx.true, yy.true, lwd=2)
points(x, rep(0,n), col=cc)
legend(6,0.22,c("Truth","Estimate"),col=c("black","red"), lty=c(1,2), bty="n")


#### Example of an EM algorithm for fitting a mixtures of K p-variate Gaussian components
#### The algorithm is tested using simulated data

## Clear the environment and load required libraries
rm(list=ls())
library(mvtnorm)    # Multivariate normals are not default in R
library(ellipse)    # Required for plotting
set.seed(63252)     # For reproducibility

## Generate data from a mixture with 3 components
KK      = 3
p       = 2
w.true = c(0.5,0.3,0.2)  # True weights associated with the components
mu.true     = array(0, dim=c(KK,p))
mu.true[1,] = c(0,0)   #True mean for the first component
mu.true[2,] = c(5,5)   #True mean for the second component
mu.true[3,] = c(-3,7)   #True mean for the third component
Sigma.true      = array(0, dim=c(KK,p,p))
Sigma.true[1,,] = matrix(c(1,0,0,1),p,p)   #True variance for the first component
Sigma.true[2,,] = matrix(c(2,0.9,0.9,1),p,p)   #True variance for the second component
Sigma.true[3,,] = matrix(c(1,-0.9,-0.9,4),p,p)   #True variance for the third component
n  = 120
cc = sample(1:3, n, replace=T, prob=w.true)
x  = array(0, dim=c(n,p))
for(i in 1:n){
  x[i,] = rmvnorm(1, mu.true[cc[i],], Sigma.true[cc[i],,])
}

par(mfrow=c(1,1))
plot(x[,1], x[,2], col=cc, type="n", xlab=expression(x[1]), ylab=expression(x[2]))
text(x[,1], x[,2], seq(1,n), col=cc, cex=0.6)
for(k in 1:KK){
  lines(ellipse(x=Sigma.true[k,,], centre=mu.true[k,], level=0.50), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma.true[k,,], centre=mu.true[k,], level=0.82), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma.true[k,,], centre=mu.true[k,], level=0.95), col="grey", lty=2, lwd=2)
}
#title(main="Data + True Components")


### Run the EM algorithm
## Initialize the parameters
w   = rep(1,KK)/KK  #Assign equal weight to each component to start with
mu  = rmvnorm(KK, apply(x,2,mean), var(x))   #RandomCluster centers randomly spread over the support of the data
Sigma      = array(0, dim=c(KK,p,p))  #Initial variances are assumed to be the same
Sigma[1,,] = var(x)/KK  
Sigma[2,,] = var(x)/KK
Sigma[3,,] = var(x)/KK

par(mfrow=c(1,1))
plot(x[,1], x[,2], col=cc, xlab=expression(x[1]), ylab=expression(x[2]))
for(k in 1:KK){
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.50), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.82), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.95), col="grey", lty=2, lwd=2)
}
title(main="Initial estimate + Observations")

s       = 0
sw      = FALSE
QQ      = -Inf
QQ.out  = NULL
epsilon = 10^(-5)

while(!sw){
  ## E step
  v = array(0, dim=c(n,KK))
  for(k in 1:KK){
    v[,k] = log(w[k]) + dmvnorm(x, mu[k,], Sigma[k,,],log=TRUE)  #Compute the log of the weights
  }
  for(i in 1:n){
    v[i,] = exp(v[i,] - max(v[i,]))/sum(exp(v[i,] - max(v[i,])))  #Go from logs to actual weights in a numerically stable manner
  }
  
  ## M step
  w     = apply(v,2,mean)
  mu    = array(0, dim=c(KK, p))
  for(k in 1:KK){
    for(i in 1:n){
      mu[k,]    = mu[k,] + v[i,k]*x[i,]
    }
    mu[k,] = mu[k,]/sum(v[,k])
  }
  Sigma = array(0, dim=c(KK, p, p))
  for(k in 1:KK){
    for(i in 1:n){
      Sigma[k,,] = Sigma[k,,] + v[i,k]*(x[i,] - mu[k,])%*%t(x[i,] - mu[k,])
    }
    Sigma[k,,] = Sigma[k,,]/sum(v[,k])
  }
  
  ##Check convergence
  QQn = 0
  for(i in 1:n){
    for(k in 1:KK){
      QQn = QQn + v[i,k]*(log(w[k]) + dmvnorm(x[i,],mu[k,],Sigma[k,,],log=TRUE))
    }
  }
  if(abs(QQn-QQ)/abs(QQn)<epsilon){
    sw=TRUE
  }
  QQ = QQn
  QQ.out = c(QQ.out, QQ)
  s = s + 1
  print(paste(s, QQn))
  
  #Plot current components over data
  layout(matrix(c(1,2),2,1), widths=c(1,1), heights=c(1.3,3))
  par(mar=c(3.1,4.1,0.5,0.5))
  plot(QQ.out[1:s],type="l", xlim=c(1,max(10,s)), las=1, ylab="Q")
  
  par(mar=c(5,4,1,0.5))
  plot(x[,1], x[,2], col=cc, main=paste("s =",s,"   Q =", round(QQ.out[s],4)), 
       xlab=expression(x[1]), ylab=expression(x[2]), lwd=2)
  for(k in 1:KK){
    lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.50), col="grey", lty=2, lwd=2)
    lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.82), col="grey", lty=2, lwd=2)
    lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.95), col="grey", lty=2, lwd=2)
  }
}

#Plot current components over data
layout(matrix(c(1,2),2,1), widths=c(1,1), heights=c(1.3,3))
par(mar=c(3.1,4.1,0.5,0.5))
plot(QQ.out[1:s],type="l", xlim=c(1,max(10,s)), las=1, ylab="Q", lwd=2)

par(mar=c(5,4,1,0.5))
plot(x[,1], x[,2], col=cc, main=paste("s =",s,"   Q =", round(QQ.out[s],4)), xlab=expression(x[1]), ylab=expression(x[2]))
for(k in 1:KK){
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.50), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.82), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.95), col="grey", lty=2, lwd=2)
}


############################################ MCMC ############################################
#### Example of an MCMC algorithm for fitting a location mixture of 2 Gaussian components
#### The algorithm is tested using simulated data

## Clear the environment and load required libraries
rm(list=ls())
library(MCMCpack)
set.seed(81196)  # So that results are reproducible

## Generate data from a mixture with 2 components
KK         = 2
w.true     = 0.6  # True weights associated with the components
mu.true    = rep(0, KK)
mu.true[1] = 0   # True mean for the first component
mu.true[2] = 5   # True mean for the second component
sigma.true = 1   # True standard deviation of all components
n          = 120         # Number of observations to be generated
cc.true = sample(1:KK, n, replace=T, prob=c(w.true,1-w.true))
x  = rep(0, n)
for(i in 1:n){
  x[i] = rnorm(1, mu.true[cc.true[i]], sigma.true)
}

# Plot the true density
par(mfrow=c(1,1))
xx.true = seq(-8,11,length=200)
yy.true = w.true*dnorm(xx.true, mu.true[1], sigma.true) + 
  (1-w.true)*dnorm(xx.true, mu.true[2], sigma.true) 
plot(xx.true, yy.true, type="l", xlab="x", ylab="True density", lwd=2)
points(x, rep(0,n), col=cc.true)

## Initialize the parameters
w     = 1/2                         #Assign equal weight to each component to start with
mu    = rnorm(KK, mean(x), sd(x))   #Random cluster centers randomly spread over the support of the data
sigma = sd(x)                       #Initial standard deviation

# Plot the initial guess for the density
xx = seq(-8,11,length=200)
yy = w*dnorm(xx, mu[1], sigma) + (1-w)*dnorm(xx, mu[2], sigma)
plot(xx, yy, type="l", ylim=c(0, max(yy)), xlab="x", 
     ylab="Initial density", lwd=2)
points(x, rep(0,n), col=cc.true)

## The actual MCMC algorithm starts here
# Priors
aa  = rep(1,KK)  # Uniform prior on w
eta = 0          # Mean 0 for the prior on mu_k
tau = 5          # Standard deviation 5 on the prior for mu_l
dd  = 2
qq  = 1

# Number of iterations of the sampler
rrr   = 6000
burn  = 1000


# Storing the samples
cc.out    = array(0, dim=c(rrr, n))
w.out     = rep(0, rrr)
mu.out    = array(0, dim=c(rrr, KK))
sigma.out = rep(0, rrr)
logpost   = rep(0, rrr)

# MCMC iterations
for(s in 1:rrr){
  # Sample the indicators
  cc = rep(0,n)
  for(i in 1:n){
    v = rep(0,KK)
    v[1] = log(w) + dnorm(x[i], mu[1], sigma, log=TRUE)  #Compute the log of the weights
    v[2] = log(1-w) + dnorm(x[i], mu[2], sigma, log=TRUE)  #Compute the log of the weights
    v = exp(v - max(v))/sum(exp(v - max(v)))
    cc[i] = sample(1:KK, 1, replace=TRUE, prob=v)
  }
  
  # Sample the weights
  w = rbeta(1, aa[1] + sum(cc==1), aa[2] + sum(cc==2))

  # Sample the means
  for(k in 1:KK){
    nk    = sum(cc==k)
    xsumk = sum(x[cc==k])
    tau2.hat = 1/(nk/sigma^2 + 1/tau^2)
    mu.hat  = tau2.hat*(xsumk/sigma^2 + eta/tau^2)
    mu[k]   = rnorm(1, mu.hat, sqrt(tau2.hat))
  }

  # Sample the variances
  dd.star = dd + n/2
  qq.star = qq + sum((x - mu[cc])^2)/2
  sigma = sqrt(rinvgamma(1, dd.star, qq.star))

  # Store samples
  cc.out[s,]   = cc
  w.out[s]     = w
  mu.out[s,]   = mu
  sigma.out[s] = sigma
  for(i in 1:n){
    if(cc[i]==1){
      logpost[s] = logpost[s] + log(w) + dnorm(x[i], mu[1], sigma, log=TRUE)
    }else{
      logpost[s] = logpost[s] + log(1-w) + dnorm(x[i], mu[2], sigma, log=TRUE)
    }
  }
  logpost[s] = logpost[s] + dbeta(w, aa[1], aa[2],log = T)
  for(k in 1:KK){
    logpost[s] = logpost[s] + dnorm(mu[k], eta, tau, log = T)
  }
  logpost[s] = logpost[s] + log(dinvgamma(sigma^2, dd, 1/qq))
  if(s/500==floor(s/500)){
    print(paste("s =",s))
  }
}

## Plot the logposterior distribution for various samples
par(mfrow=c(1,1))
par(mar=c(4,4,1,1)+0.1)
plot(logpost, type="l", xlab="Iterations", ylab="Log posterior")

xx = seq(-8,11,length=200)
density.posterior = array(0, dim=c(rrr-burn,length(xx)))
for(s in 1:(rrr-burn)){
  density.posterior[s,] = density.posterior[s,] + w.out[s+burn]*dnorm(xx,mu.out[s+burn,1],sigma.out[s+burn]) +
                                                  (1-w.out[s+burn])*dnorm(xx,mu.out[s+burn,2],sigma.out[s+burn])
}
density.posterior.m = apply(density.posterior , 2, mean)
density.posterior.lq = apply(density.posterior, 2, quantile, 0.025)
density.posterior.uq = apply(density.posterior, 2, quantile, 0.975)
par(mfrow=c(1,1))
par(mar=c(4,4,1,1)+0.1)
plot(xx, density.posterior.m, type="n",ylim=c(0,max(density.posterior.uq)), xlab="x", ylab="Density")
polygon(c(xx,rev(xx)), c(density.posterior.lq, rev(density.posterior.uq)), col="grey", border="grey")
lines(xx, density.posterior.m, lwd=2)
points(x, rep(0,n), col=cc.true)

#### Example of an MCMC algorithm for fitting a mixtures of K p-variate Gaussian components
#### The algorithm is tested using simulated data

## Clear the environment and load required libraries
rm(list=ls())
library(mvtnorm)
library(ellipse)
library(MCMCpack)

## Generate data from a mixture with 3 components
KK      = 3
p       = 2
w.true = c(0.5,0.3,0.2)  # True weights associated with the components
mu.true     = array(0, dim=c(KK,p))
mu.true[1,] = c(0,0)   #True mean for the first component
mu.true[2,] = c(5,5)   #True mean for the second component
mu.true[3,] = c(-3,7)   #True mean for the third component
Sigma.true      = array(0, dim=c(KK,p,p))
Sigma.true[1,,] = matrix(c(1,0,0,1),p,p)   #True variance for the first component
Sigma.true[2,,] = matrix(c(2,0.9,0.9,1),p,p)   #True variance for the second component
Sigma.true[3,,] = matrix(c(1,-0.9,-0.9,4),p,p)   #True variance for the third component
set.seed(63252)    #Keep seed the same so that we can reproduce results
n  = 120
cc.true = sample(1:3, n, replace=T, prob=w.true)
x  = array(0, dim=c(n,p))
for(i in 1:n){
  x[i,] = rmvnorm(1, mu.true[cc.true[i],], Sigma.true[cc.true[i],,])
}

par(mfrow=c(1,1))
plot(x[,1], x[,2], col=cc.true, xlab=expression(x[1]), ylab=expression(x[2]), type="n")
text(x[,1], x[,2], seq(1,n), col=cc.true, cex=0.6)
for(k in 1:KK){
  lines(ellipse(x=Sigma.true[k,,], centre=mu.true[k,], level=0.50), col="grey", lty=2)
  lines(ellipse(x=Sigma.true[k,,], centre=mu.true[k,], level=0.82), col="grey", lty=2)
  lines(ellipse(x=Sigma.true[k,,], centre=mu.true[k,], level=0.95), col="grey", lty=2)
}
title(main="Data + True Components")

## Initialize the parameters
w          = rep(1,KK)/KK  #Assign equal weight to each component to start with
mu         = rmvnorm(KK, apply(x,2,mean), var(x))   #RandomCluster centers randomly spread over the support of the data
Sigma      = array(0, dim=c(KK,p,p))  #Initial variances are assumed to be the same
Sigma[1,,] = var(x)/KK  
Sigma[2,,] = var(x)/KK
Sigma[3,,] = var(x)/KK
cc         = sample(1:KK, n, replace=TRUE, prob=w)

par(mfrow=c(1,1))
plot(x[,1], x[,2], col=cc.true, xlab=expression(x[1]), ylab=expression(x[2]))
for(k in 1:KK){
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.50), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.82), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.95), col="grey", lty=2, lwd=2)
}
title(main="Initial estimate + Observations")

# Priors
aa = rep(1, KK)
dd = apply(x,2,mean)
DD = 10*var(x)
nu = p
SS = var(x)/3

# Number of iteration of the sampler
rrr = 1000

# Storing the samples
cc.out    = array(0, dim=c(rrr, n))
w.out     = array(0, dim=c(rrr, KK))
mu.out    = array(0, dim=c(rrr, KK, p))
Sigma.out = array(0, dim=c(rrr, KK, p, p))
logpost   = rep(0, rrr)

for(s in 1:rrr){
  # Sample the indicators
  for(i in 1:n){
    v = rep(0,KK)
    for(k in 1:KK){
      v[k] = log(w[k]) + mvtnorm::dmvnorm(x[i,], mu[k,], Sigma[k,,], log=TRUE)  #Compute the log of the weights
    }
    v = exp(v - max(v))/sum(exp(v - max(v)))
    cc[i] = sample(1:KK, 1, replace=TRUE, prob=v)
  }
  
  # Sample the weights
  w = as.vector(rdirichlet(1, aa + tabulate(cc)))
  
  # Sample the means
  DD.st = matrix(0, nrow=p, ncol=p)
  for(k in 1:KK){
    mk    = sum(cc==k)
    xsumk = apply(x[cc==k,], 2, sum)
    DD.st = solve(mk*solve(Sigma[k,,]) + solve(DD))
    dd.st = DD.st%*%(solve(Sigma[k,,])%*%xsumk + solve(DD)%*%dd)
    mu[k,] = as.vector(rmvnorm(1,dd.st,DD.st))
  }
  
  # Sample the variances
  xcensumk = array(0, dim=c(KK,p,p))
  for(i in 1:n){
    xcensumk[cc[i],,] = xcensumk[cc[i],,] + (x[i,] - mu[cc[i],])%*%t(x[i,] - mu[cc[i],])
  }
  for(k in 1:KK){
    Sigma[k,,] = riwish(nu + sum(cc==k), SS + xcensumk[k,,])
  }
  
  # Store samples
  cc.out[s,]      = cc
  w.out[s,]       = w
  mu.out[s,,]     = mu
  Sigma.out[s,,,] = Sigma
  for(i in 1:n){
    logpost[s] = logpost[s] + log(w[cc[i]]) + mvtnorm::dmvnorm(x[i,], mu[cc[i],], Sigma[cc[i],,], log=TRUE)
  }
  logpost[s] = logpost[s] + ddirichlet(w, aa)
  for(k in 1:KK){
    logpost[s] = logpost[s] + mvtnorm::dmvnorm(mu[k,], dd, DD, log=TRUE)
    logpost[s] = logpost[s] + log(diwish(Sigma[k,,], nu, SS))
  }
  
  if(s/250==floor(s/250)){
    print(paste("s = ", s))
  }
  
}

## Plot the logposterior distribution for various samples
par(mfrow=c(1,1))
par(mar=c(4,4,1,1)+0.1)
plot(logpost, type="l", xlab="Iterations", ylab="Log posterior")

## Plot the density estimate for the last iteration of the MCMC
par(mfrow=c(1,1))
par(mar=c(4,4,2,1)+0.1)
plot(x[,1], x[,2], col=cc.true, main=paste("s =",s,"   logpost =", round(logpost[s],4)), xlab=expression(x[1]), ylab=expression(x[2]))
for(k in 1:KK){
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.50), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.82), col="grey", lty=2, lwd=2)
  lines(ellipse(x=Sigma[k,,], centre=mu[k,], level=0.95), col="grey", lty=2, lwd=2)
}

