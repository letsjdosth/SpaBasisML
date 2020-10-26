rm(list=ls())
library(fields);library(classInt);library(glmnet);#library(nimble)
plotRF <- function(dat,rangeDat,label,location,length.out=10){
  breaks <- seq(range(rangeDat,na.rm = TRUE)[1],
                range(rangeDat,na.rm = TRUE)[2],
                length.out=length.out)
  pal <- tim.colors(length(breaks)-1)
  fb <- classIntervals(dat, n = length(pal),
                       style = "fixed", fixedBreaks = breaks)
  col <- findColours(fb, pal)
  plot(x=location[,1],y=location[,2],col=col, pch=16,
       main=label)
}

# Basis functions representations
# Mod : data
# Rank : # of knots
# CV: for cross validation
set.seed(12345)
nRank <- 100; nMod <- 1000; nCV <- floor(nMod*0.5)
indRank <- 1:nRank ; indMod <- (nRank+1):(nRank+nMod) ; indCV <- (nRank+nMod+1):(nRank+nMod+nCV)
gridLocationRank <- as.matrix(expand.grid(seq(0,1,length.out = sqrt(nRank)),seq(0,1,length.out = sqrt(nRank))))
gridLocationMod <- cbind(runif(nMod),runif(nMod))
gridLocationCV <- cbind(runif(nCV),runif(nCV))

# spatial covariate 
XMat <- cbind(runif(nMod),runif(nMod))
XMatCV <- cbind(runif(nCV),runif(nCV))
XMatRank <- cbind(runif(nRank),runif(nRank))
XMatComplete <- rbind(XMatRank,XMat,XMatCV)

# Generate Data
phi <- 0.2; sigma2 <- 1; beta <- c(2,2)
gridLocationComplete <- rbind(gridLocationRank,gridLocationMod,gridLocationCV)
distMat <- as.matrix(rdist(gridLocationComplete))
covMat <- (1+(sqrt(5)*(distMat/phi))+((5*distMat^2)/(3*(phi^2))))*exp(-(sqrt(5)*(distMat/phi)))
# covMat<-exp(-(distMat/phi))
covMat <- sigma2*covMat
TSseed <- rnorm(nRank+nMod+nCV) # Matern type cov
w <- t(chol(covMat))%*%TSseed
z <- sapply(exp(w+XMatComplete%*%beta),rpois,n=1)
#z=pois(exp(w+Xb)), w~N(0,1)*MaternCov

# thin plate splinte basis
tpsMat <- (distMat[indMod,indRank]^2)*log(distMat[indMod,indRank])  
tpsMatCV <- (distMat[indCV,indRank]^2)*log(distMat[indCV,indRank])  

# design matrix
DesignMat <- cbind(XMat,tpsMat)
DesignMatCV <- cbind(XMatCV,tpsMatCV)


# Data plots
par(mfrow=c(2,3),mar=c(2,2,2,2))
plotRF(dat=z,rangeDat=z,label="Obs",location=gridLocationComplete,length.out=10)
plotRF(dat=exp(w+gridLocationComplete%*%beta),rangeDat=z,label="Expected",location=gridLocationComplete,length.out=20)
plotRF(dat=w,rangeDat=w,label="RE",location=gridLocationComplete,length.out=20)

plotRF(dat=w[indRank],rangeDat=w[indRank],label="Knots",location=gridLocationComplete[indRank,],length.out=20)
plotRF(dat=w[indMod],rangeDat=w[indMod],label="Model",location=gridLocationComplete[indMod,],length.out=20)
plotRF(dat=w[indCV],rangeDat=w[indCV],label="Validation",location=gridLocationComplete[indCV,],length.out=20)


##### Fitting GLM 100 seconds for 1000 points #####


# glm fits without penalties
glmfit <- glm(z[indMod]~ -1+DesignMat, family="poisson")
summary(glmfit)

par(mfrow=c(2,4))
plotRF(dat=w[indMod],rangeDat=w,label="True RE",location=gridLocationComplete[indMod,],length.out=20)
plotRF(dat=w[indCV],rangeDat=w,label="CV RE",location=gridLocationComplete[indCV,],length.out=20)
plotRF(dat=z[indMod],rangeDat=z,label="Obs",location=gridLocationComplete[indMod,],length.out=20)
plotRF(dat=z[indCV],rangeDat=z,label="CV Response",location=gridLocationComplete[indCV,],length.out=20)

plotRF(dat=as.numeric(cbind(tpsMat)%*%glmfit$coef[-c(1,2)]),rangeDat=w,label="Estimated RE",location=gridLocationComplete[indMod,],length.out=20)
plotRF(dat=as.numeric(cbind(tpsMatCV%*%glmfit$coef[-c(1,2)])),rangeDat=w,label="Predicted RE",location=gridLocationComplete[indCV,],length.out=20)
plotRF(dat=as.numeric(exp(DesignMat%*%glmfit$coef)),rangeDat=z,label="Estimated Intensity",location=gridLocationComplete[indMod,],length.out=20)
plotRF(dat=as.numeric(exp(DesignMatCV%*%glmfit$coef)),rangeDat=z,label="Predicted Intensity",location=gridLocationComplete[indCV,],length.out=20)


