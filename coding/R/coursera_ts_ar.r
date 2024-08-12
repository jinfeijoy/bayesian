####################################################
#####             MLE for AR(1)               ######
####################################################
set.seed(2021)
phi=0.9 # ar coefficient
v=1
sd=sqrt(v) # innovation standard deviation
T=500 # number of time points
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

## Case 1: Conditional likelihood
y=as.matrix(yt[2:T]) # response
X=as.matrix(yt[1:(T-1)]) # design matrix
phi_MLE=as.numeric((t(X)%*%y)/sum(X^2)) # MLE for phi
s2=sum((y - phi_MLE*X)^2)/(length(y) - 1) # Unbiased estimate for v 
v_MLE=s2*(length(y)-1)/(length(y)) # MLE for v

cat("\n MLE of conditional likelihood for phi: ", phi_MLE, "\n",
    "MLE for the variance v: ", v_MLE, "\n", 
    "Estimate s2 for the variance v: ", s2, "\n")

####################################################
#####             MLE for AR(1)               ######
####################################################
set.seed(2021)
phi=0.9 # ar coefficient
sd=1 # innovation standard deviation
T=200 # number of time points
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) # sample stationary AR(1) process

y=as.matrix(yt[2:T]) # response
X=as.matrix(yt[1:(T-1)]) # design matrix
phi_MLE=as.numeric((t(X)%*%y)/sum(X^2)) # MLE for phi
s2=sum((y - phi_MLE*X)^2)/(length(y) - 1) # Unbiased estimate for v
v_MLE=s2*(length(y)-1)/(length(y)) # MLE for v 

print(c(phi_MLE,s2))

#######################################################
######     Posterior inference, AR(1)               ###
######     Conditional Likelihood + Reference Prior ###
######     Direct sampling                          ###
#######################################################

n_sample=3000   # posterior sample size

## step 1: sample posterior distribution of v from inverse gamma distribution
v_sample=1/rgamma(n_sample, (T-2)/2, sum((yt[2:T] - phi_MLE*yt[1:(T-1)])^2)/2)

## step 2: sample posterior distribution of phi from normal distribution
phi_sample=rep(0,n_sample)
for (i in 1:n_sample){
phi_sample[i]=rnorm(1, mean = phi_MLE, sd=sqrt(v_sample[i]/sum(yt[1:(T-1)]^2)))}

## plot histogram of posterior samples of phi and v
par(mfrow = c(1, 2), cex.lab = 1.3)
hist(phi_sample, xlab = bquote(phi), 
     main = bquote("Posterior for "~phi),xlim=c(0.75,1.05), col='lightblue')
abline(v = phi, col = 'red')
hist(v_sample, xlab = bquote(v), col='lightblue', main = bquote("Posterior for "~v))
abline(v = sd, col = 'red')


###########################################################
##### Compute AR reciprocal roots given the AR coefficients
###########################################################
# Assume the folloing AR coefficients for an AR(8)
phi=c(0.27, 0.07, -0.13, -0.15, -0.11, -0.15, -0.23, -0.14)
roots=1/polyroot(c(1, -phi)) # compute reciprocal characteristic roots
r=Mod(roots) # compute moduli of reciprocal roots
lambda=2*pi/Arg(roots) # compute periods of reciprocal roots
# print results modulus and frequency by decreasing order
print(cbind(r, abs(lambda))[order(r, decreasing=TRUE), ][c(2,4,6,8),]) 


###########################################################
##### Simulating data from an AR(p)
###########################################################
## simulate data from an AR(2)
set.seed(2021)
## AR(2) with a pair of complex-valued roots with modulus 0.95 and period 12 
r=0.95
lambda=12 
phi=numeric(2) 
phi[1]<- 2*r*cos(2*pi/lambda) 
phi[2] <- -r^2
phi
T=300 # number of time points
sd=1 # innovation standard deviation
yt=arima.sim(n=T, model = list(ar = phi), sd=sd)

par(mfrow = c(3, 1), cex.lab = 1.5)
## plot simulated data 
ts.plot(yt)
## draw sample autocorrelation function
acf(yt, lag.max = 50,
    type = "correlation", ylab = "sample ACF", 
    lty = 1, ylim = c(-1, 1), main = " ")

## draw sample partial autocorrelation function
pacf(yt, lag.ma = 50, main = "sample PACF")

###########################################################
##### Maximum likelihood estimation, AR(p), conditional likelihood
###########################################################
set.seed(2021)
# Simulate 300 observations from an AR(2) with one pair of complex-valued reciprocal roots 
r=0.95
lambda=12 
phi=numeric(2) 
phi[1]=2*r*cos(2*pi/lambda) 
phi[2]=-r^2
sd=1 # innovation standard deviation
T=300 # number of time points
# generate stationary AR(2) process
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

## Compute the MLE for phi and the unbiased estimator for v using the conditional likelihood
p=2
y=rev(yt[(p+1):T]) # response
X=t(matrix(yt[rev(rep((1:p),T-p)+rep((0:(T-p-1)),rep(p,T-p)))],p,T-p));
XtX=t(X)%*%X
XtX_inv=solve(XtX)
phi_MLE=XtX_inv%*%t(X)%*%y # MLE for phi
s2=sum((y - X%*%phi_MLE)^2)/(length(y) - p) #unbiased estimate for v

cat("\n MLE of conditional likelihood for phi: ", phi_MLE, "\n",
    "Estimate for v: ", s2, "\n")

###########################################################
##### Bayesian inference, AR(p), conditional likelihood
###########################################################
# Simulate 300 observations from an AR(2) with one pair of complex-valued roots 
set.seed(2021)
r=0.95
lambda=12 
phi=numeric(2) 
phi[1]=2*r*cos(2*pi/lambda) 
phi[2]=-r^2
sd=1 # innovation standard deviation
T=300 # number of time points
# generate stationary AR(2) process
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 
par(mfrow=c(1,1))
plot(yt)

## Compute the MLE of phi and the unbiased estimator of v using the conditional likelihood
p=2
y=rev(yt[(p+1):T]) # response
X=t(matrix(yt[rev(rep((1:p),T-p)+rep((0:(T-p-1)),rep(p,T-p)))],p,T-p));
XtX=t(X)%*%X
XtX_inv=solve(XtX)
phi_MLE=XtX_inv%*%t(X)%*%y # MLE for phi
s2=sum((y - X%*%phi_MLE)^2)/(length(y) - p) #unbiased estimate for v

#####################################################################################
### Posterior inference, conditional likelihood + reference prior via 
### direct sampling                 
#####################################################################################

n_sample=1000 # posterior sample size
library(MASS)

## step 1: sample v from inverse gamma distribution
v_sample=1/rgamma(n_sample, (T-2*p)/2, sum((y-X%*%phi_MLE)^2)/2)

## step 2: sample phi conditional on v from normal distribution
phi_sample=matrix(0, nrow = n_sample, ncol = p)
for(i in 1:n_sample){
  phi_sample[i, ]=mvrnorm(1,phi_MLE,Sigma=v_sample[i]*XtX_inv)
}

par(mfrow = c(2, 3), cex.lab = 1.3)
## plot histogram of posterior samples of phi and v

for(i in 1:2){
  hist(phi_sample[, i], xlab = bquote(phi), 
       main = bquote("Histogram of "~phi[.(i)]),col='lightblue')
  abline(v = phi[i], col = 'red')
}

hist(v_sample, xlab = bquote(nu), main = bquote("Histogram of "~v),col='lightblue')
abline(v = sd, col = 'red')

#####################################################
# Graph posterior for modulus and period 
#####################################################
r_sample=sqrt(-phi_sample[,2])
lambda_sample=2*pi/acos(phi_sample[,1]/(2*r_sample))
hist(r_sample,xlab="modulus",main="",col='lightblue')
abline(v=0.95,col='red')
hist(lambda_sample,xlab="period",main="",col='lightblue')
abline(v=12,col='red')


###################################################
# Model order selection 
###################################################
# Simulate data from an AR(2)
set.seed(2021)
r=0.95
lambda=12 
phi=numeric(2) 
phi[1]=2*r*cos(2*pi/lambda) 
phi[2]=-r^2
sd=1 # innovation standard deviation
T=300 # number of time points
# generate stationary AR(2) process
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

#############################################################################
######   compute AIC and BIC for different AR(p)s based on simulated data ###
#############################################################################
pmax=10 # the maximum of model order
Xall=t(matrix(yt[rev(rep((1:pmax),T-pmax)+rep((0:(T-pmax-1)),
              rep(pmax,T-pmax)))], pmax, T-pmax));
y=rev(yt[(pmax+1):T])
n_cond=length(y) # (number of total time points - the maximum of model order)

## compute MLE
my_MLE <- function(y, Xall, p){
  n=length(y)
  x=Xall[,1:p]
  a=solve(t(x) %*%x)
  a=(a + t(a))/2 # for numerical stability 
  b=a%*%t(x)%*%y # mle for ar coefficients
  r=y - x%*%b # residuals 
  nu=n - p # degrees freedom
  R=sum(r*r) # SSE
  s=R/nu #MSE
  return(list(b = b, s = s, R = R, nu = nu))
}


## function for AIC and BIC computation 
AIC_BIC <- function(y, Xall, p){
  ## number of time points
  n <- length(y)
  
  ## compute MLE
  tmp=my_MLE(y, Xall, p)
  
  ## retrieve results
  R=tmp$R
  
  ## compute likelihood
  likl= n*log(R)
  
  ## compute AIC and BIC
  aic =likl + 2*(p)
  bic =likl + log(n)*(p)
  return(list(aic = aic, bic = bic))
}
# Compute AIC, BIC 
aic =numeric(pmax)
bic =numeric(pmax)

for(p in 1:pmax){
  tmp =AIC_BIC(y,Xall, p)
  aic[p] =tmp$aic
  bic[p] =tmp$bic
  print(c(p, aic[p], bic[p])) # print AIC and BIC by model order
}

## compute difference between the value and its minimum
aic =aic-min(aic) 
bic =bic-min(bic) 

## draw plot of AIC, BIC, and the marginal likelihood
par(mfrow = c(1, 1))
matplot(1:pmax,matrix(c(aic,bic),pmax,2),ylab='value',
        xlab='AR order p',pch="ab", col = 'black', main = "AIC and BIC")
# highlight the model order selected by AIC
text(which.min(aic), aic[which.min(aic)], "a", col = 'red') 
# highlight the model order selected by BIC
text(which.min(bic), bic[which.min(bic)], "b", col = 'red') 

########################################################
p <- which.min(bic) # We set up the moder order
print(paste0("The chosen model order by BIC: ", p))

###################################################
# Spectral density of AR(p)
###################################################
### Simulate 300 observations from an AR(2) prcess with a pair of complex-valued roots 
set.seed(2021)
r=0.95
lambda=12 
phi=numeric(2) 
phi[1]<- 2*r*cos(2*pi/lambda) 
phi[2] <- -r^2
sd=1 # innovation standard deviation
T=300 # number of time points
# sample from the AR(2) process
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

# Compute the MLE of phi and the unbiased estimator of v using the conditional likelihood 
p=2
y=rev(yt[(p+1):T])
X=t(matrix(yt[rev(rep((1:p),T-p)+rep((0:(T-p-1)),rep(p,T-p)))],p,T-p));
XtX=t(X)%*%X
XtX_inv=solve(XtX)
phi_MLE=XtX_inv%*%t(X)%*%y # MLE for phi
s2=sum((y - X%*%phi_MLE)^2)/(length(y) - p) #unbiased estimate for v

# Obtain 200 samples from the posterior distribution under the conditional likelihood and the reference prior 
n_sample=200 # posterior sample size
library(MASS)

## step 1: sample v from inverse gamma distribution
v_sample=1/rgamma(n_sample, (T-2*p)/2, sum((y-X%*%phi_MLE)^2)/2)

## step 2: sample phi conditional on v from normal distribution
phi_sample=matrix(0, nrow = n_sample, ncol = p)
for(i in 1:n_sample){
  phi_sample[i,]=mvrnorm(1,phi_MLE,Sigma=v_sample[i]*XtX_inv)
}


### using spec.ar to draw spectral density based on the data assuming an AR(2)
spec.ar(yt, order = 2, main = "yt")

### using arma.spec from astsa package to draw spectral density
library("astsa")

## plot spectral density of simulated data with posterior sampled 
## ar coefficients and innvovation variance
par(mfrow = c(1, 1))
result_MLE=arma.spec(ar=phi_MLE, var.noise = s2, log='yes',main = '')
freq=result_MLE$freq
  
spec=matrix(0,nrow=n_sample,ncol=length(freq))
for (i in 1:n_sample){
result=arma.spec(ar=phi_sample[i,], var.noise = v_sample[i], log='yes',
                 main = '')
spec[i,]=result$spec
}

plot(2*pi*freq,log(spec[1,]),type='l',ylim=c(-3,12),ylab="log spectra",
     xlab="frequency",col=0)
for (i in 1:n_sample){
lines(2*pi*freq,log(spec[i,]),col='darkgray')
}
lines(2*pi*freq,log(result_MLE$spec))
abline(v=2*pi/12,lty=2,col='red')

