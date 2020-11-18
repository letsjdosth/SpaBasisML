######  Non-separable space time cov simulation studies  ######
rm(list=ls())
library(fields);library(classInt);library(glmnet);#library(nimble)
#setwd("C:/Users/admin/Dropbox/Nonstationary/Code/BenCode/previousBenCodes/LASSO_Spatial_Ben")
#source("sharedFunctions.R")
plotRF<-function(dat,rangeDat,label,location,length.out=10,type=1){
  if(type==1){
  breaks <- seq(range(rangeDat,na.rm = TRUE)[1],
                range(rangeDat,na.rm = TRUE)[2],
                length.out=length.out)
  }else{
  breaks <- seq(range(rangeDat,na.rm = TRUE)[1],
                quantile(rangeDat,prob=0.99),
                length.out=length.out)
  }
  pal <- tim.colors(length(breaks)-1)
  fb <- classIntervals(dat, n = length(pal),
                       style = "fixed", fixedBreaks = breaks)
  col <- findColours(fb, pal)
  plot(x=location[,1],y=location[,2],col=col, pch=16,
       main=label)
}

# non-separable covariance function eq(6.1) in DNNGP paper
covfn <- function(dist.t, dist.s, s2, a, c, k){ 
      K = s2/((a*dist.t^2 + 1)^(k))*exp( -c*dist.s/((a*dist.t^2 + 1)^(k/2)) ) 
return(K)
}

##### data simulation for 3 different scenarios #####
# Datta et al.(2016) Nonseparable dynamic nearest neighbor Gaussian process models for large spatio-temporal data with an application to particulate matter analysis." 
# 1) short spatial range/long temporal range
# 2) long spatial/temporal range
# 3) long spatial/short temporal range
set.seed(3)
s2 = 1      # overall variance
a = 500     # temporal corr (50, 500, 2000)
c = 2.5     # spatial corr  (25, 2.5, 2.5)
k = 0.5     # space-time decay (0.75, 0.5, 0.95)
p.true <- c(s2, a, c, k) # true cov parameters
beta <- c(1,1)  # regression coefficients 

nTime <- 8             # number of time 
nMod <- 600           # number of observed spatial observation
nCV <- floor(nMod*0.5) # number of unobserved spatial observation

gridLocationMod <- cbind(runif(nMod),runif(nMod))  # simulate data locations
gridLocationCV <- cbind(runif(nCV),runif(nCV))     # unobserved data locations
gridLocationData <- rbind(gridLocationMod, gridLocationCV)

coords <- gridLocationData
for(i in 1:(nTime-1)){ coords <- rbind(coords,gridLocationData) }  # same locations across time 
dist.s <- rdist(coords, coords)                                                                                            # compute spatial distance matrix
dist.t <- rdist( rep(seq(0,1,length.out = nTime),each = nMod + nCV), rep(seq(0,1,length.out = nTime),each = nMod + nCV) )  # compute temporal distance


# spatio-temporal variable
XMat <- cbind(runif(nMod*nTime),runif(nMod*nTime))
XMatCV <- cbind(runif(nCV*nTime),runif(nCV*nTime))
XMatComplete <- rbind(XMat,XMatCV)

covMat <- covfn(dist.t, dist.s, p.true[1], p.true[2], p.true[3], p.true[4])  # compute covariance matrix
TSseed <- rnorm((nMod + nCV)*nTime)                                        # simulate random effects
w <- t(chol(covMat))%*%TSseed
z <- sapply(exp(w+XMatComplete%*%beta),rpois,n=1)

par(mfrow=c(2,4),mar=c(2,2,2,2))
for(i in 1:nTime){ plotRF(dat=w[(1+(i-1)*(nMod + nCV)):(i*(nMod + nCV))],rangeDat=w,label="RE",location=gridLocationData,length.out=20,type=1) }
par(mfrow=c(2,4),mar=c(2,2,2,2))
for(i in 1:nTime){ plotRF(dat=z[(1+(i-1)*(nMod + nCV)):(i*(nMod + nCV))],rangeDat=z,label="RE",location=gridLocationData,length.out=20,type=1) }
 

##### Basis functions representations #####
nRank <- 100 # number of knots for space
gridLocationRank <- as.matrix(expand.grid(seq(0,1,length.out = sqrt(nRank)),seq(0,1,length.out = sqrt(nRank))))
indRank<-1:nRank ; indMod<-(nRank+1):(nRank+nMod) ; indCV<-(nRank+nMod+1):(nRank+nMod+nCV)

gridLocationComplete <- rbind(gridLocationRank,gridLocationData)
distMatSpace <- as.matrix(rdist(gridLocationComplete))

# thin plate spline basis for space 
tpsMat <- (distMatSpace[indMod,indRank]^2)*log(distMatSpace[indMod,indRank])  
tpsMatCV <- (distMatSpace[indCV,indRank]^2)*log(distMatSpace[indCV,indRank])  

# Harmonics basis for time 
nTime
timeVect <- 1:nTime
j <- seq(1,nTime/2)
freqW <- j/nTime
basisFunctionA <- sapply(freqW,function(x){cos(2*pi*x*timeVect)})
basisFunctionB <- sapply(freqW,function(x){sin(2*pi*x*timeVect)})
basisFunction <- matrix(NA,nrow=nrow(basisFunctionB),ncol=2*ncol(basisFunctionB))

for(i in 1:(nTime/2)){
  basisFunction[,2*(i-1)+1] <- basisFunctionA[,i]
  basisFunction[,2*(i-1)+2] <- basisFunctionB[,i]
}
plot(x=timeVect,y=basisFunction[,1],typ="n",ylim=range(basisFunction))
for(i in 1:8){
  lines(x=timeVect,y=basisFunction[,i],col=i)
}

#  Spatio-Temporal Basis using thin plate spline
stBasis <- kronecker(basisFunction,tpsMat)
stBasisCV <- kronecker(basisFunction,tpsMatCV)
dim(stBasis);dim(stBasisCV)

# design matrix
DesignMat <- as.matrix(cbind(XMat,stBasis))
DesignMatCV <- as.matrix(cbind(XMatCV,stBasisCV))

###### glm fits without penalties
indObs <- 1:nMod 
for(i in 2:nTime){ indObs <- c(indObs, ((nMod+nCV)*(i-1)+1):((nMod+nCV)*(i-1)+nMod)) }

glmfit <- glm(z[indObs]~ -1+DesignMat, family="poisson")
summary(glmfit)
glmfit$coef[1:2]
wobs <- w[indObs]
wBasis <- stBasis%*%glmfit$coef[-c(1,2)]

# random effect comparison
par(mfrow=c(4,4),mar=c(2,2,2,2))
for(i in 1:nTime){ plotRF(dat=wobs[(1+(i-1)*nMod):(i*nMod)],rangeDat=w,label="RE",location=gridLocationMod,length.out=20) }
for(i in 1:nTime){ plotRF(dat=wBasis[(1+(i-1)*nMod):(i*nMod)],rangeDat=w,label="RE",location=gridLocationMod,length.out=20) }

par(mfrow=c(1,3))
plot(w[indObs],wBasis );abline(0,1)
plot(XMat%*%beta, XMat%*%glmfit$coef[1:2]);abline(0,1)
plot(XMat%*%beta + w[indObs], XMat%*%glmfit$coef[1:2]+wBasis);abline(0,1)

true = XMat%*%beta + w[indObs]
est = XMat%*%glmfit$coef[1:2]+wBasis

par(mfrow=c(4,4),mar=c(2,2,2,2))
for(i in 1:nTime){ plotRF(dat=true[(1+(i-1)*nMod):(i*nMod)],rangeDat=true,label="RE",location=gridLocationMod,length.out=20) }
for(i in 1:nTime){ plotRF(dat=est[(1+(i-1)*nMod):(i*nMod)],rangeDat=true,label="RE",location=gridLocationMod,length.out=20) }



