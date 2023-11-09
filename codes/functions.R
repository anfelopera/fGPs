library(fields)
library(DiceKriging)
library(orthopolynom)


#' compute matrix of wasserstein distance between distributions in set 1 and 2
#'
#' @param mq1 matrix of the quantile values for equispaced inputs in (0,1) for the distributions in set 1.
#' Matrix of size n1*d, with n1 the number of distributions and d the number of discretization points
#' @param mq2 same for set 2. matrix of size n2*d
#'         
#' @return the n1*n2 matrix of the wasserstein distances between the distributions from set 1 and set 2
#'
#' @examples wasserstein_matrix_quantile( rbind(seq(from=0,to=1,by=0.001),seq(from=0,to=1,by=0.001)^200),rbind(seq(from=0,to=1,by=0.001),seq(from=0,to=1,by=0.001)^200 ))
wasserstein_matrix_quantile = function( mq1 , mq2) {
  d = dim(mq1)[2]
  rdist(mq1,mq2)/sqrt(d)
}


#' compute a cross-covariance or covariance matrix with the exponential wasserstein covariance function
#'
#' @param sigma: variance
#' @param lc: correlation length
#' @param mq1 matrix of the quantile values for equispaced inputs in (0,1) for the distributions in set 1.
#' Matrix of size n1*d, with n1 the number of distributions and d the number of discretization points
#' @param  mq2 same for set 2. matrix of size n2*d
#'
#' @return the covariance or cross-covariance matrix of size n1*n2
#'
#' @examples cov_matrix_exp_quantile( 2 , 3.4 , 1,  rbind(seq(from=0,to=1,by=0.001),seq(from=0,to=1,by=0.001)^200),rbind(seq(from=0,to=1,by=0.001),seq(from=0,to=1,by=0.001)^200 ))
cov_matrix_exp_quantile = function (sigma,lc,H,mq1,mq2) {
  sigma^2 * exp( - wasserstein_matrix_quantile(mq1,mq2)^(2*H)  / lc)
}


#' Prediction by simple kriging on a vector of new locations
#'
#' @param R_obs: covariance matrix of observations
#' @param R_obs_pred: cross covariance matrix between observations and prediction locations 
#' @param y_obs: observation vector 
#' @param nugget: add to the covariance matrix of the observations: nugget*largest diagonal element*identity 
#'
#' @return vector of predictions
#'
#' @examples
prediction = function(R_obs,R_obs_pred,y_obs,nugget) {
  R_obs = R_obs+max(diag(R_obs))*nugget*diag(dim(R_obs)[1])
  t(R_obs_pred)%*%solve(R_obs)%*%y_obs
}

#' Predictive variance by simple kriging on a vector of new locations
#'
#' @param R_obs: covariance matrix of observations
#' @param R_red: covariance matrix at prediction locations
#' @param R_obs_pred: cross covariance matrix between observations and prediction locations 
#' @param nugget: add to the covariance matrix of the observations: nugget*largest diagonal element*identity 
#'
#' @return vector of predictive variance
#'
#' @examples
predictive_sd = function(R_obs,R_pred,R_obs_pred,nugget) {
  R_obs = R_obs+max(diag(R_obs))*nugget*diag(dim(R_obs)[1])
  sqrt(diag( R_pred -  t(R_obs_pred)%*%solve(R_obs)%*%R_obs_pred ))
}


#' Compute -2*log(likelihood) for a gaussian vector with mean zero
#'
#' @param R: covariance matrix 
#' @param y_obs: observation vector
#' @param nugget: add to the covariance matrix of the observations: nugget*largest diagonal element*identity 
#'
#' @return the modified log likelihood value
#'
#' @examples
modified_log_likelihood = function(R,y_obs,nugget) {
  n = length(y_obs)
  R = R+max(diag(R))*nugget*diag(dim(R)[1])
  cR = chol(R)
  iR <- chol2inv(cR)
  2*sum(log(diag(cR)))+sum( y_obs * ( iR %*% y_obs) )
}


#' maximum likelihood estimation (one gradient descent)
#'
#' @param init_theta: starting point of the gradient descent
#' @param theta_inf: vector of lower bound for the optimization 
#' @param theta_sup: vector of upper bound for the optimization 
#' @param ind_active: vector of indices of covariance parameters w.r.t. which we optimize (others are left to starting point value)
#' @param nugget: numerical nugget variance for matrix inversion
#' @param cov_matrix_function: the covariance matrix function used. takes inputs (theta,mq)
#' @param mq_obs: n*d matrix of quantiles values for input points. n= number of inputs, d=number of quantile values  
#' @param y_obs: vector of the observations of size n 
#'
#' @return $hat_theta: the vector hat_theta of the maximum likelihood estimator
#' @return $val: the value of the modified log likelihood at the optimizer
#'
#' @examples
maximum_likelihood_descent = function(init_theta,theta_inf,theta_sup,ind_active,nugget,cov_matrix_function,mq_obs,y_obs)  {
  
  to_min <-function( param_active , ind_active , init_param , nugget, cov_matrix_function , mq_obs, y_obs )  {
    param = init_param
    param[ind_active] = param_active
    R = cov_matrix_function(param,mq_obs) 
    modified_log_likelihood(R,y_obs,nugget) 
  }
  #
  #gradient descent
  res_opt = optim( par=init_theta[ind_active] , f=to_min , method="L-BFGS-B" , lower = theta_inf , upper = theta_sup,
                  ind_active = ind_active, init_param  = init_theta, nugget = nugget, 
                  cov_matrix_function = cov_matrix_function, mq_obs = mq_obs, y_obs=y_obs)
  hat_param = init_theta
  hat_param[ind_active] = res_opt$par
  list(hat_theta=hat_param,val=res_opt$value)
}

#' maximum likelihood estimation (best of k gradient descent)
#'
#' @param init_theta: starting point of the gradient descent
#' @param theta_inf: vector of lower bound for the optimization 
#' @param theta_sup: vector of upper bound for the optimization 
#' @param ind_active: vector of indices of covariance parameters w.r.t. which we optimize (others are left to starting point value)
#' @param nugget: numerical nugget variance for matrix inversion
#' @param cov_matrix_function: the covariance matrix function used. takes inputs (theta,mq)
#' @param mq_obs: n*d matrix of quantiles values for input points. n= number of inputs, d=number of quantile values  
#' @param y_obs: vector of the observations of size n 
#' @param k: number of gradient descent 
#'
#' @return $hat_theta: the vector hat_theta of the maximum likelihood estimator
#' @return $val: the criterion value at the maximum likelihood estimator
#' @return $m_hat_theta: the matrix of the hat_theta for the different gradient descents
#' @return $v_val: the vector of the optimum values for the different gradient descents
#'
#' @examples
maximum_likelihood = function(init_theta,theta_inf,theta_sup,ind_active,nugget,cov_matrix_function,mq_obs,y_obs,k)  {
  m_hat_theta = matrix(nrow=k,ncol=length(init_theta),data=NA)
  v_val = seq(from=0,to=0,length=k)
  for (i in 1:k) {
    init_theta[ind_active] = theta_inf[ind_active] + runif(n=length(theta_inf[ind_active]),min=0,max=1)*
      (theta_sup[ind_active]-theta_inf[ind_active])
    res_opt = maximum_likelihood_descent(init_theta,theta_inf,theta_sup,ind_active,nugget,cov_matrix_function,mq_obs,y_obs)
    m_hat_theta[i,] = res_opt$hat_theta
    v_val[i] = res_opt$val
  }
  hat_theta = m_hat_theta[which.min(v_val),]
  val = min(v_val)
  list(m_hat_theta=m_hat_theta,v_val=v_val,hat_theta=hat_theta,val=val)
}



#' generate a matrix of random pdf functions on [0,1]. each pdf is obtained by normalizing the exponential of
#' a gaussian process on [0,1], with Matern covariance function
#'
#' @param param: vector with components: 
#' sigma: standard deviation of the gaussian process  
#' lc: correlation length of the gaussian process 
#' nu: smoothness parameter of the gaussian process (1/2, 3/2 or 5/2)
#' thereshold: the pdf values are set to 0 if they are lower than threshold. Then threshold is removed to
#' @param n: number of pdf to be generated 
#' @param d: number of discretization points for representing the pdf
#' the other pdf values. Then the pdf is normalized again.
#'
#' @return the n*d matrix of the generated pdf
generate_pdf_matern = function(n,d,param) {
  sigma = param[1]
  lc = param[2]
  nu = param[3]
  threshold = param[4]
  M = matrix(nrow=n,ncol=d,data=NA)
  if (nu==1/2) {
    cov_type= "exp"
  }
  if (nu==3/2) {
    cov_type= "matern3_2"
  }
  if (nu==3/2) {
    cov_type= "matern5_2"
  }
  for (i in 1:n) {
    x = seq(from=0,to=1,length=d)
    km = km(formula = ~1,design = x,response=x, covtype= cov_type , coef.trend = 0, coef.cov = lc, coef.var = sigma^2)
    v = as.vector(simulate(object=km,nsim=1,cond=FALSE))
    ev = exp(v)
    ev = ev/mean(ev)
    ev=ev-threshold
    ev[ev<=0] = 0
    ev = ev/mean(ev)
    M[i,] = ev
  }
  M
}


#' convert a matrix of discretized pdf into a matrix of discretized quantiles
#'
#' @param m_pdf: n*d matrix of n pdf curves on (0,1), discretized at d equispaced points of (0,1)
#'
#' @return n*d matrix of n the corresponding n quantile curves on (0,1), discretized at d equispaced points of (0,1)
#'
#' @examples
from_pdf_to_quantile = function(m_pdf) {
  n = dim(m_pdf)[1]
  d = dim(m_pdf)[2]
  m_q = m_pdf
  for (i in 1:n) {
    v = (1/sum(m_pdf[i,]))*cumsum(m_pdf[i,])
    w = v
    for (j in 1:d) {
      ind = seq(from=1,to=d,length=d)
      w[j] = min(ind[v[ind]>=j/d])
    }
    w = w/d
    w[w==Inf] = 1
    m_q[i,] = w
  }
  m_q
}

RMSE = function(y,hat_y) {
  sqrt(mean((y-hat_y)^2))
}

Q2 = function(y,hat_y) {
  1 - RMSE(y,hat_y)^2/var(y)
}

log_predictive_probability = function(y,hat_y,hat_sigma) {
  mean( (1/2)*log(2*pi*hat_sigma^2) + ((y-hat_y)^2)/(2*hat_sigma^2) )
}

confidence_interval_coverage = function(y,hat_y,hat_sigma,alpha) {
  mean( abs(y-hat_y) <= qnorm(p=1-(1-alpha)/2,mean=0,sd=1)*hat_sigma )
}

#' compute the matrix of coefficients of the projection of a matrix of pdf on the basis of the Legendre polynomials
#'
#' @param m_pdf: n*d matrix of the n pdf (defined on [0,1]) on an equispaced grid of size d
#' @param order 
#'
#' @return matrix of projection coefficients 0,...,order - 1
#'
#' @examples
from_pdf_to_vector_legendre = function(m_pdf,order) {
  n=dim(m_pdf)[1]
  m_proj = matrix(nrow=n,ncol=order)
  for (i in 1:n) {
    m_proj[i,] = projection_legendre(m_pdf[i,],order-1)
  }
  m_proj
}

#' compute the matrix of coefficients of the projection of a matrix of pdf on a PCA basis
#'
#' @param m_pdf: n*d matrix of the n pdf (defined on [0,1]) on an equispaced grid of size d
#' @param order: number of PCA projections 
#'
#' @return m_proj: matrix of projection coefficients 1,...,order, of size n*order
#' @return m_component: matrix of size d*d of the principal component functions (by decreasing order of importance) 
#' (a line = a component)
from_pdf_to_vector_PCA = function(m_pdf,order) {
  n=dim(m_pdf)[1]
  m_component = t(eigen(t(m_pdf)%*%m_pdf)$vectors)
  m_component = m_component[1:order,]
  for (i in 1:order) {
    m_component[i,] = m_component[i,] / sqrt(mean(m_component[i,]^2))
  }
  m_proj = (1/d) * m_pdf %*% t( m_component )
  list(m_proj = m_proj , m_component = m_component)
}

#' compute the coefficients of the projection of a function on the basis of the Legendre polynomials
#'
#' @param v  values of the function to project (defined on [0,1]) on an equispaced grid
#' 
#' @param order 
#'
#' @return vector of projection coefficients 0,...,order
#'
#' @examples which( projection_legendre(  sqrt(3)*(2*((1:100)/100) - 1) , 4 ) > 0.9 ) == 2
projection_legendre = function(v,order)  {
  t = seq(from=0,to=1,length.out = length(v))
  alpha = array(NaN,order+1)
  poly = legendre.polynomials(order, normalized=TRUE)
  for (i in 0:order) {
    v_poly = sqrt(2) * predict(poly[[i+1]],2*t-1)
    alpha[i+1] = mean(v_poly*v)
  }
  alpha
}


f_code_m_sd = function(v_pdf) {
  x = seq(from=0,to=1,length=length(v_pdf))
  m = mean( v_pdf * x )
  sd_ = sqrt( mean( v_pdf * x^2 ) - m^2 )
  m / (0.05 + sd_)
} 

f_code_cos = function(v_pdf) {
  x = seq(from=0,to=1,length=length(v_pdf))
  m = mean( v_pdf * cos( 5*x ) )
  m^2 - m + exp(m)-1
} 

#' sample gaussian densities truncated on [0,1]
#'
#' @param n: number of densities 
#' @param d: number of equispaced discretization points 
#' @param param: vector with lower and upper bound for mean and lower and upper bound
#' for standard deviation
#'
#' @return: n*d matrix of the generated densities
generate_pdf_m_sd = function(n,d,param) {
  m_min = param[1]
  m_max = param[2]
  sd_min = param[3]
  sd_max = param[4]
  M = matrix(nrow=n,ncol=d,data=NA)
  for (i in 1:n) {
    m = runif(n=1,min=m_min,max=m_max)
    sd_ = runif(n=1,min=sd_min,max=sd_max)
    M[i,] = dnorm(x=seq(from=0,to=1,length=d),mean=m,sd=sd_)
    M[i,] = M[i,] / mean(M[i,])
  }
  M
}


#' sample gaussian densities truncated on [0,1] with random perturbation
#'
#' @param n: number of densities 
#' @param d: number of equispaced discretization points 
#' @param param: vector with lower and upper bound for mean and lower and upper bound
#' for standard deviation; covariance parameters of the random perturbation
#'
#' @return: n*d matrix of the generated densities
generate_pdf_m_sd_pert = function(n,d,param) {
  m_min = param[1]
  m_max = param[2]
  sd_min = param[3]
  sd_max = param[4]
  sd_noise = param[5]
  lc = param[6]
  nu = param[7]
  M = matrix(nrow=n,ncol=d,data=NA)
  for (i in 1:n) {
    m = runif(n=1,min=m_min,max=m_max)
    sd_ = runif(n=1,min=sd_min,max=sd_max)
    M[i,] = dnorm(x=seq(from=0,to=1,length=d),mean=m,sd=sd_)
    x = seq(from=0,to=1,length=d)
    if (nu==1/2) {
      cov_type= "exp"
    }
    if (nu==3/2) {
      cov_type= "matern3_2"
    }
    if (nu==5/2) {
      cov_type= "matern5_2"
    }
    km = km(formula = ~1,design = x,response=x, covtype= cov_type , coef.trend = 0, coef.cov = lc, coef.var = sd_noise^2)
    v = as.vector(simulate(object=km,nsim=1,cond=FALSE))
    M[i,] = M[i,] * exp(v)
    M[i,] = M[i,] / mean(M[i,])
  }
  M
}

tri_kernel = function(vx,h) {
  #compute values of the triangular kernel (1-abs(x/h)) 1{  abs(x/h) <=1 } 
  #INPUTS
  #vx: vector of differences
  #h: bandwidth
  #OUTPUTS
  #vector of kernel values
  (1-abs(vx/h)) * ( abs(vx/h) <= 1)
} 

density_estimate = function(vlearn,vtest,h) {
  #compute estimated density at test points from learning points 
  #INPUTS
  #vlearn: learning vector of sampled values 
  #vtest: vector of test values
  #h: bandwidth
  #OUTPUTS
  #vector of density estimates
  mabsdiff = abs(outer(vlearn,vtest,'-'))/h
  mker = (1/h) * (1-mabsdiff) * ( mabsdiff <= 1)
  (1/length(vlearn)) * colSums( mker )
} 

select_h_density_estimate = function(vtraining,vvalidation,nh) {
  #select h by maximizing the log likelihood on validation set
  #INPUTS
  #vtraining: training vector of sampled values (support points for density evaluation)
  #vvalidation: vector of validation values (where log likelihood is maximized)
  #nh: number of h values over which we maximize
  #OUTPUTS
  #h: bandwidth maximizing log likelihood
  vh = runif(n=nh)
  vloglik = rep(-1,nh)
  for (i in 1:nh) {   #loop on h values
    vloglik[i] = sum(log( density_estimate(vtraining,vvalidation,vh[i]) ))
  }
  vh[which.max(vloglik)]
} 



kernel_regression = function(mlearn,ylearn,mtest,h,b,nDistEval) {
  #compute predictions at test samples from learning samples and values 
  #INPUTS
  #mlearn: n*m matrix: line i is a m-sample from learning distribution i  
  #ylearn: vector of size n of the outputs for each input distribution
  #mtest: ntest*m matrix: line i is a m-sample from test distribution i
  #h: bandwidth for kernel regression from estimated distances
  #b: bandwidth for density estimation
  #nDistEval: number of points for integral distance evaluations
  #OUTPUTS
  #vector of predicted values
  nlearn = dim(mlearn)[1]
  ntest = dim(mtest)[1]
  xdens = seq(from = -0.2,to=1.2,length=nDistEval)
  mDist = matrix(nrow=nlearn,ncol=ntest,data=-1)
  mdenslearn = matrix(nrow=nlearn,ncol=nDistEval,data=-1)
  mdenstest = matrix(nrow=ntest,ncol=nDistEval,data=-1)
  for (i in 1:nlearn) { 
    mdenslearn[i,] = density_estimate(vlearn = mlearn[i,],vtest=xdens,h=b)
  }
  for (i in 1:ntest) { 
    mdenstest[i,] = density_estimate(vlearn = mtest[i,],vtest=xdens,h=b)
  }
  for (i in 1:nlearn) {
    for (j in 1:ntest) {
      vdenslearn = mdenslearn[i,]
      vdenstest = mdenstest[j,]
      mDist[i,j] = mean(abs(vdenslearn-vdenstest))
    }
  }
  vpred = rep(-1,dim(mtest)[1])
  for (i in 1:ntest) {   #loop on test points
    if ( sum( tri_kernel(mDist[,i],h) )  >0  ) {
      vpred[i] = sum( ylearn * tri_kernel(mDist[,i],h) ) / sum( tri_kernel(mDist[,i],h) )
    }
    else {
      vpred[i] = mean(ylearn)
    }
  }
  vpred
} 

select_bh_kernel_regression = function(mtraining,ytraining,mvalidation,yvalidation,nDistEval,nbh,hlim,blim) {
  #select b and h by minimizing the rmse on the validation set 
  #INPUTS
  #mtraining: n*m matrix: line i is a m-sample from training distribution i  
  #ytraining: vector of size n of the training outputs for each training distribution
  #mvalidation: nval*m matrix: line i is a m-sample from validation distribution i
  #yvalidation: vector of size nval of the validation outputs for each validation distribution
  #nbh: number of paris (b,h) over which we minimize rmse
  #nDistEval: number of points for integral distance evaluations
  #hlim: bounds for h
  #blim: bounds for b
  #OUTPUTS
  #vector of predicted values
  vh = runif(n=nbh,min = hlim[1],max=hlim[2])
  vb = runif(n=nbh,min = blim[1],max=blim[2])
  mbh = cbind(vb,vh)
  vrmse = rep(-1,nbh)
  for (i in 1:nbh) {   #loop on pairs (b,h)
    cat("i=",i,"\n")
    pred = kernel_regression(mlearn=mtraining,ylearn=ytraining,mtest=mvalidation,
                            h=mbh[i,2],b=mbh[i,1],nDistEval)
    vrmse[i] = sqrt(mean((pred-yvalidation  )^2))
  }
  mbh[which.min(vrmse),]
} 
