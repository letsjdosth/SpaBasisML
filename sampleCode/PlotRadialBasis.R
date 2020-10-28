library(fields);library(classInt);#library(INLA)
# Generate Time Series
rm(list=ls())
#### 
plotRF<-function(dat,rangeDat,label,location,length.out=10){
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
set.seed(123456)
nRank= 25; nMod<-1000 ; nCV<-floor(nMod*0.2)
indRank<-1:nRank ; indMod<-(nRank+1):(nRank+nMod) ; indCV<-(nRank+nMod+1):(nRank+nMod+nCV)
gridLocationRank<-as.matrix(expand.grid(seq(0,1,length.out = sqrt(nRank)),seq(0,1,length.out = sqrt(nRank))))
gridLocationMod<-cbind(runif(nMod),runif(nMod))
gridLocationCV<-cbind(runif(nCV),runif(nCV))
# Generate Data
phi=0.2; sigma2=1
gridLocationComplete<-rbind(gridLocationRank,gridLocationMod,gridLocationCV)
distMat<-as.matrix(rdist(gridLocationComplete))
covMat<-(1+(sqrt(5)*(distMat/phi))+((5*distMat^2)/(3*(phi^2))))*exp(-(sqrt(5)*(distMat/phi)))
covMat<-exp(-(distMat/phi))
# covMat<-sigma2*covMat
TSseed<-rnorm(nRank+nMod+nCV)
w<-t(chol(covMat))%*%TSseed
z<-sapply(exp(w),rpois,n=1)

############################################
#dev.off()
par(mfrow=c(2,3),mar=c(2,2,2,2))
plotRF(dat=z,rangeDat=z,label="Obs",location=gridLocationComplete,length.out=10)
plotRF(dat=exp(w),rangeDat=exp(w),label="Expected",location=gridLocationComplete,length.out=20)
plotRF(dat=w,rangeDat=w,label="RE",location=gridLocationComplete,length.out=20)

plotRF(dat=w[indRank],rangeDat=w[indRank],label="Knots",location=gridLocationComplete[indRank,],length.out=20)
plotRF(dat=w[indMod],rangeDat=w[indMod],label="Model",location=gridLocationComplete[indMod,],length.out=20)
plotRF(dat=w[indCV],rangeDat=w[indCV],label="Validation",location=gridLocationComplete[indCV,],length.out=20)


# Distance
distMat<-as.matrix(dist(gridLocationComplete))
distBasis<-distMat[c(indMod),indRank]
dim(distBasis)

####################################################################################################################################


# GAussian
phi<-3
gaussBasis<-function(phi,dist){
  exp(-(phi*dist)^2)
}
basisMat<-apply(distBasis,2,gaussBasis,phi=phi)
par(mfrow=c(6,6),mar=c(2,2,2,2))
apply(basisMat,2,plotRF,rangeDat=basisMat,label="Gauss",location=gridLocationMod,length.out=10)
# plotRF(dat=basisMat[,1],rangeDat=basisMat,label="Gauss",location=gridLocationMod,length.out=10)

# Multi-quadratic
phi<-200
multiQuadBasis<-function(phi,dist){
  sqrt(1+(phi*dist)^2)
}
basisMat<-apply(distBasis,2,multiQuadBasis,phi=phi)
par(mfrow=c(6,6),mar=c(2,2,2,2))
apply(basisMat,2,plotRF,rangeDat=basisMat,label="Gauss",location=gridLocationMod,length.out=10)

# Inverse Quadratic
phi<-1
invQuadBasis<-function(phi,dist){
  1/(1+(phi*dist)^2)
}
basisMat<-apply(distBasis,2,invQuadBasis,phi=phi)
par(mfrow=c(6,6),mar=c(2,2,2,2))
apply(basisMat,2,plotRF,rangeDat=basisMat,label="Gauss",location=gridLocationMod,length.out=10)


# Inverse multiquadratic
phi<-1
invMultQuadBasis<-function(phi,dist){
  1/sqrt(1+(phi*dist)^2)
}
basisMat<-apply(distBasis,2,invMultQuadBasis,phi=phi)
par(mfrow=c(6,6),mar=c(2,2,2,2))
apply(basisMat,2,plotRF,rangeDat=basisMat,label="Gauss",location=gridLocationMod,length.out=10)

#고쳤음
# Thin-plate spline
tpsBasis<-function(dist){
  (dist)*log(dist)
}
basisMat<-apply(distBasis,2,tpsBasis)
par(mfrow=c(6,6),mar=c(2,2,2,2))
apply(basisMat,1,plotRF,rangeDat=basisMat,label="Gauss",location=gridLocationRank,length.out=10)

# Bump Functions

# Polyharmonic



