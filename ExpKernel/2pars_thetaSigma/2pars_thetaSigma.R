library(NeuralEstimators)
library(JuliaConnectoR)
juliaEval("using NeuralEstimators, Flux, 
          GraphNeuralNetworks, Statistics") # Load Julia packages for GNN functionality

library(ggplot2)
library(latex2exp)

library(parallel) # mclapply()
library(spatstat) # rMatClust()


# Simulate from GP --------------------------------------------------------

# Simulate spatial locations by a Matern cluster process
simulateS <- function(K) {
  # Simulate spatial configurations over the unit square, with each configuration
  # drawn from a Matern cluster process with different, randomly sampled hyperparameters
  n      <- sample(100:300, K, replace = TRUE)  # approximate sample size 
  lambda <- runif(K, 10, 50)                    # intensity of parent Poisson point process
  mu     <- n/lambda                            # mean number of daughter points 
  r      <- runif(K, 0.05, 0.2)                 # cluster radius
  
  S <- lapply(1:K, function(k) {
    pts <- rMatClust(lambda[k], r = r[k], mu = mu[k])
    cbind(x = pts$x, y = pts$y)
  })
  
  return(S)
}

# Simulate observed values at locations S by GP with exponential kernel
simulateZ <- function(theta, S, m = 1) {
  # Simulate conditionally independent replicates for each pair of 
  # parameters and spatial configurations
  Z <- mclapply(1:ncol(theta), function(k) {
    D <- as.matrix(dist(S[[k]])) # distance matrix   
    n <- nrow(D)                 # sample size
    Sigma <- exp(-D/theta[k])    # covariance matrix
    L <- t(chol(Sigma))          # Cholesky factor of the covariance matrix
    e <- matrix(rnorm(n*m), nrow = n, ncol = m) # standard normal variates
    Z <- L %*% e                 # conditionally independent replicates from the model
    Z
  })
  
  return(Z)
}

theta <- matrix(c(0.05, 0.1, 0.2, 0.4), nrow = 1)
K     <- ncol(theta)
S     <- simulateS(K)
Z     <- simulateZ(theta, S)

df <- Map(function(z, th, s) {
  data.frame(Z = c(z), theta = th, s1 = s[, 1], s2 = s[, 2])
}, Z, theta, S)
df <- do.call(rbind, df)

df$theta <- paste0("$\\theta$ = ", round(df$theta, 2))
df$theta <- as.factor(df$theta)
levels(df$theta) <- sapply(levels(df$theta), TeX)

ggplot(df) +
  geom_point(aes(x = s1, y = s2, colour = Z)) +
  facet_grid(~theta, labeller = label_parsed) +
  scale_colour_viridis_c(option = "magma") +
  labs(x = expression(s[1]), y = expression(s[2])) +
  scale_x_continuous(breaks = c(0.2, 0.5, 0.8)) +
  scale_y_continuous(breaks = c(0.2, 0.5, 0.8)) +
  coord_fixed() +
  theme_bw()


# GNN ---------------------------------------------------------------------

# Sampling from the prior 
# K: number of samples to draw from the prior
sampler <- function(K) { 
  theta <- runif(K, max = 0.4) # draw from the prior 
  theta <- t(theta)            # reshape to matrix
  return(theta)
}

simulate <- function(theta, S, ...) {
  K <- ncol(theta)
  S <- simulateS(K)
  Z <- simulateZ(theta, S, ...)
  G <- spatialgraph(S, Z) # graph from the spatial configurations and associated spatial data
  return(G)
}

Time1 = Sys.time()

K <- 500
theta_train <- sampler(K)
theta_val   <- sampler(K/10)
Z_train <- simulate(theta_train)
Z_val   <- simulate(theta_val)

# Initialise the estimator 
estimator <- juliaEval('

  # Spatial weight functions: continuous surrogates for 0-1 basis functions 
  h_max = 0.15 # maximum distance to consider 
  q = 10       # output dimension of the spatial weights
  w = KernelWeights(h_max, q)
  
  # Propagation module
  propagation = GNNChain(
    SpatialGraphConv(1 => q, relu, w = w, w_out = q),
    SpatialGraphConv(q => q, relu, w = w, w_out = q)
  )
  
  # Readout module
  readout = GlobalPool(mean)
  
  # Inner network
  psi = GNNSummary(propagation, readout)
  
  # Expert summary statistics, the empirical variogram
  V = NeighbourhoodVariogram(h_max, q)
  
  # Outer network
  phi = Chain(
    Dense(2q  => 128, relu), 
    Dense(128 => 128, relu), 
    Dense(128 => 1, softplus) # softplus activation to ensure positive estimates
  )
  
  # DeepSet object 
  deepset = DeepSet(psi, phi; S = V)
  
  # Point estimator
  estimator = PointEstimator(deepset)
')

# Train the estimator
estimator <- train(
  estimator,
  theta_train = theta_train,
  theta_val   = theta_val,
  Z_train     = Z_train,
  Z_val       = Z_val,
  epochs      = 20,
  stopping_epochs = 3
)

# Assess the estimator
theta_test <- sampler(1000)
Z_test     <- simulate(theta_test)
assessment <- assess(estimator, theta_test, Z_test, estimator_names = "GNN NBE")
#>  Running GNN NBE...
png("./assessPlot.png")
plotestimates(assessment)
dev.off()

Time2 = Sys.time()
print(Time2 - Time1)
