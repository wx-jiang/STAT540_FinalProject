library(NeuralEstimators)
library(JuliaConnectoR)
juliaEval("using NeuralEstimators, Flux, 
          GraphNeuralNetworks, Statistics") # Load Julia packages for GNN functionality

library(ggplot2)
library(patchwork)
library(latex2exp)

library(parallel) # mclapply()
library(spatstat) # rMatClust()

library(rstan)

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
    Sigma <- theta[2,k] * exp(-D/theta[1,k])    # covariance matrix
    L <- t(chol(Sigma))          # Cholesky factor of the covariance matrix
    e <- matrix(rnorm(n*m), nrow = n, ncol = m) # standard normal variates
    Z <- L %*% e                 # conditionally independent replicates from the model
    Z
  })
  
  return(Z)
}

## Illustration of the simulations

# theta <- matrix(c(0.05, 0.1, 0.2, 0.4,
#                   0.01, 0.05, 0.1, 0.2),
#                 byrow = T,
#                 nrow = 2, 
#                 ncol = 4)
# K     <- ncol(theta)
# S     <- simulateS(K)
# Z     <- simulateZ(theta, S)
# 
# df <- Map(function(z, phi, sigma, s) {
#   data.frame(Z = c(z), phi = phi, sigma = sigma, s1 = s[, 1], s2 = s[, 2])
# }, Z, theta[1,], theta[2,], S)
# df <- do.call(rbind, df)
# 
# df$phi <- paste0("$\\phi$ = ", round(df$phi, 2))
# df$phi <- as.factor(df$phi)
# levels(df$phi) <- sapply(levels(df$phi), TeX)
# 
# df$sigma <- paste0("$\\sigma$ = ", round(df$sigma, 2))
# df$sigma <- as.factor(df$sigma)
# levels(df$sigma) <- sapply(levels(df$sigma), TeX)
# 
# ggplot(df) +
#   geom_point(aes(x = s1, y = s2, colour = Z)) +
#   facet_grid(~phi + sigma, labeller = label_parsed) +
#   scale_colour_viridis_c(option = "magma") +
#   labs(x = expression(s[1]), y = expression(s[2])) +
#   scale_x_continuous(breaks = c(0.2, 0.5, 0.8)) +
#   scale_y_continuous(breaks = c(0.2, 0.5, 0.8)) +
#   coord_fixed() +
#   theme_bw()

# GNN ---------------------------------------------------------------------

# Sampling from the prior 
# K: number of samples to draw from the prior
sampler <- function(K) {
  phi <- runif(K, max = 0.4) 
  sigma <- runif(K, max = 0.4) # draw from the prior 
  theta <- rbind(phi, sigma)   # reshape to matrix
  return(theta)
}

simulate <- function(theta, S, ...) {
  K <- ncol(theta)
  S <- simulateS(K)
  Z <- simulateZ(theta, S, ...)
  G <- spatialgraph(S, Z) # graph from the spatial configurations and associated spatial data
  return(G)
}

#### Testing setup
# Number of assessment
n_reps <- 100
theta_test = sampler(n_reps)
K_test <- ncol(theta_test)
S_test <- simulateS(K_test)
Z_test <- simulateZ(theta_test, S_test, 1)
Z_GNN_test <- spatialgraph(S_test, Z_test)
####

GNN_time1 = Sys.time()

K <- 5000
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
    Dense(128 => 2, softplus) # softplus activation to ensure positive estimates
  )
  
  # DeepSet object 
  deepset = DeepSet(psi, phi; S = V)
  
  # Point estimator
  estimator = PointEstimator(deepset)
')

# Train the estimator
point_estimator <- train(
  estimator,
  theta_train = theta_train,
  theta_val   = theta_val,
  Z_train     = Z_train,
  Z_val       = Z_val,
  epochs      = 50,
  stopping_epochs = 3
)

lower_credible_interval_estimator <- NeuralEstimators::train(
  estimator,
  loss = "
    using Statistics
    
    function mse(yhat, y; t = 0.025, agg = mean)
      agg((yhat .- y) .* (ifelse.(yhat .> y, 1.0, 0.0) .- t))
    end
  ",
  theta_train = theta_train,
  theta_val = theta_val,
  Z_train = Z_train,
  Z_val = Z_val,
  epochs = 50,
  stopping_epochs = 3
)

upper_credible_interval_estimator <- NeuralEstimators::train(
  estimator,
  loss = "
    using Statistics
    
    function mse(yhat, y; t = 0.975, agg = mean)
      agg((yhat .- y) .* (ifelse.(yhat .> y, 1.0, 0.0) .- t))
    end
  ",
  theta_train = theta_train,
  theta_val = theta_val,
  Z_train = Z_train,
  Z_val = Z_val,
  epochs = 50,
  stopping_epochs = 3
)

# Assess the estimator
assessment <- assess(point_estimator, theta_test, Z_GNN_test, estimator_names = "GNN NBE")

# MSE
point_estimate = estimate(point_estimator, Z_GNN_test)
GNN_MSE = apply((point_estimate - theta_test)^2, 1, mean)
names(GNN_MSE) = c("phi_MSE_GNN", "sigma_MSE_GNN")
print(GNN_MSE)

# Coverage probability
lower_ci = estimate(lower_credible_interval_estimator, Z_GNN_test)
upper_ci = estimate(upper_credible_interval_estimator, Z_GNN_test)
GNN_coverage_prob = apply(theta_test > lower_ci & theta_test < upper_ci, 1, mean)
names(GNN_coverage_prob) = c("phi_coverage_GNN", "sigma_coverage_GNN")
print(GNN_coverage_prob)

GNN_time2 = Sys.time()

png("./GNN.png", width = 1600, height = 800, res = 200)
plotestimates(assessment, 
              parameter_labels = c("θ1" = expression(phi),
                                   "θ2" = expression(sigma))) +
  labs(title = "GNN", subtitle = paste0("Time taken: ", format(GNN_time2 - GNN_time1)), 
       x = "Truth", y = "GNN Estimates") +
  theme(legend.position = "none")
dev.off()

# MLE ---------------------------------------------------------------------

exp_kernel <- function(S, Z, theta) {
  D <- as.matrix(dist(S[[1]])) # distance matrix   
  n <- nrow(D)                 # sample size
  Sigma <- theta[2] * exp(-D/theta[1])    # covariance matrix
  
  return(Sigma)
}

negloglik <- function(theta, S, Z) {
  theta = exp(theta)
  
  Sigma <- exp_kernel(S, Z, theta) # varcov matrix 
  Sigma <- Sigma + diag(1e-5, nrow(Sigma)) # add some jitter for numerical stability
  Sigma_inv <- solve(Sigma)
  
  # log determinant
  L <-  chol(Sigma)
  log_det <- 2 * sum(log(diag(L)))
  
  loglik <- -0.5 * log_det - 
    0.5 * t(Z[[1]]) %*% Sigma_inv %*% Z[[1]] - 
    0.5 * length(Z[[1]]) * log(2 * pi)
  
  return(-loglik)
}

mle_time1 <- Sys.time()

mle_estimates <- matrix(NA, nrow = 2, ncol = n_reps,
                        dimnames = list(c("mle_phi","mle_sigma")))

mle_se <- matrix(NA, nrow = 2, ncol = n_reps, 
                 dimnames = list(c("phi_se","sigma_se")))

for (i in seq_len(n_reps)) {
  S_i <- list( S_test[[i]] )
  Z_i <- list( Z_test[[i]] )
  
  # run optim on the log‑scale
  fit_i <- optim(
    par    = c(0, 0),
    fn     = negloglik, 
    S      = S_i, 
    Z      = Z_i, 
    lower = log(c(1e-4, 1e-4)),
    upper = log(c(1, 1)),
    method = "L-BFGS-B"
  )
  
  # back‑transform and store
  mle_estimates["mle_phi", i] <- exp(fit_i$par[1])
  mle_estimates["mle_sigma", i] <- exp(fit_i$par[2])
  
  # parametric bootstrap under the fitted model
  hat <- exp(fit_i$par)
  
  B = 50
  phi_bs   <- numeric(B)
  sigma_bs <- numeric(B)
  
  for (b in seq_len(B)) {
    Zb <- simulateZ(matrix(hat, nrow = 2), S_i, 1)
    fb <- optim(par = log(hat), fn = negloglik,
                S = S_i, Z = Zb, method = "L-BFGS-B")
    estb <- exp(fb$par)
    phi_bs[b]   <- estb[1]
    sigma_bs[b] <- estb[2]
  }
  mle_se["phi_se",i]   <- sd(phi_bs)
  mle_se["sigma_se",i] <- sd(sigma_bs)
}

# MSE
MLE_MSE = apply((mle_estimates - theta_test)^2, 1, mean)
names(MLE_MSE) = c("phi_MSE_MLE", "sigma_MSE_MLE")
print(MLE_MSE)

# Coverage probability
lower_ci = mle_estimates - 1.96 * mle_se
upper_ci = mle_estimates + 1.96 * mle_se
MLE_coverage_prob = apply(theta_test > lower_ci & theta_test < upper_ci, 1, mean)
names(MLE_coverage_prob) = c("phi_coverage_MLE", "sigma_coverage_MLE")
print(MLE_coverage_prob)

mle_time2 <- Sys.time()

mle_df <- data.frame(
  phi            = theta_test[1, ],
  sigma          = theta_test[2, ],
  mle_phi_hat    = mle_estimates["mle_phi", ],
  mle_sigma_hat  = mle_estimates["mle_sigma", ]
)

phi_plot <- ggplot(mle_df, aes(x = phi, y = mle_phi_hat)) +
  geom_point(color = "red") +
  geom_abline(linetype = 2, slope = 1, intercept = 0) +
  labs(title = expression(phi), x = "True φ", y = "MLE φ̂") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

sigma_plot <- ggplot(mle_df, aes(x = sigma, y = mle_sigma_hat)) +
  geom_point(color = "red") +
  geom_abline(linetype = 2, slope = 1, intercept = 0) +
  labs(title = expression(sigma), x = "True σ", y = "MLE σ̂") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

png("MLE.png", width = 1600, height = 800, res = 200)
phi_plot + sigma_plot +
  plot_annotation(
    title    = "MLE",
    subtitle = paste0("Time taken: ", format(mle_time2 - mle_time1))
  )
dev.off()

# RWMC --------------------------------------------------------------------

log_posterior <- function(log_theta, S, Z) {
  # unpack and back‐transform
  theta <- exp(log_theta)
  
  # build covariance (with tiny jitter for stability)
  Sigma <- exp_kernel(S, Z, theta)
  Sigma <- Sigma + diag(1e-6, nrow(Sigma))
  
  # Cholesky-based log-likelihood
  L <- chol(Sigma)
  log_det <- 2 * sum(log(diag(L)))
  z <- as.numeric(Z[[1]])            # assume single replicate
  quad <- sum(backsolve(L, z, transpose = TRUE)^2)
  n <- length(z)
  
  loglik <- -0.5 * (log_det + quad + n * log(2*pi))
  return(loglik)  # flat prior on log-θ means posterior ∝ likelihood
}

rw_metropolis <- function(log_theta_start, S, Z,
                          n_iter = 5000,
                          prop_sd = c(0.25, 0.25)) {
  chain <- matrix(NA, nrow = 2, ncol = n_iter)
  chain[, 1] <- log_theta_start
  current_lp <- log_posterior(log_theta_start, S, Z)
  # accept_tracker = 0
  
  for (i in 2:n_iter) {
    prop <- rnorm(2, mean = chain[, i-1], sd = prop_sd)
    prop_lp <- log_posterior(prop, S, Z)
    if (log(runif(1)) < (prop_lp - current_lp)) {
      # accept_tracker = accept_tracker + 1
      chain[, i] <- prop
      current_lp <- prop_lp
    } else {
      chain[, i] <- chain[, i-1]
    }
  }
  rownames(chain) <- c("log_phi", "log_sigma")
  # print(paste("Acceptance rate is", accept_tracker/n_iter))
  return(chain)
}

rw_post_means <- matrix(NA, nrow = 2, ncol = n_reps,
                        dimnames = list(c("rw_phi_mean","rw_sigma_mean")))

rw_post_lower_quantile <- matrix(NA, nrow = 2, ncol = n_reps,
                        dimnames = list(c("rw_phi_lower","rw_sigma_lower")))

rw_post_upper_quantile <- matrix(NA, nrow = 2, ncol = n_reps,
                        dimnames = list(c("rw_phi_upper","rw_sigma_upper")))

rw_time1 <- Sys.time()

init <- log(c(0.1, 0.1)) # starting values on log scale
burn_in <- 500

for (i in seq_len(n_reps)) {
  # pull in the i-th dataset
  S_i <- list(S_test[[i]])
  Z_i <- list(Z_test[[i]])
  
  # run RW Metropolis; capture printed acceptance rate
  chain <- rw_metropolis(init, S_i, Z_i)
  
  # discard burn-in
  post_chain <- chain[, (burn_in+1):5000, drop = FALSE]
  
  # back-transform and compute posterior means
  rw_post_means["rw_phi_mean",   i] <- median(exp(post_chain["log_phi",   ]))
  rw_post_means["rw_sigma_mean", i] <- median(exp(post_chain["log_sigma", ]))
  
  rw_post_lower_quantile["rw_phi_lower", i] <- quantile(exp(post_chain["log_phi",   ]), 0.025)
  rw_post_lower_quantile["rw_sigma_lower", i] <- quantile(exp(post_chain["log_sigma",   ]), 0.025) 
  
  rw_post_upper_quantile["rw_phi_upper", i] <- quantile(exp(post_chain["log_phi",   ]), 0.975)
  rw_post_upper_quantile["rw_sigma_upper", i] <- quantile(exp(post_chain["log_sigma",   ]), 0.975)
}

# MSE
RWMC_MSE = apply((rw_post_means - theta_test)^2, 1, mean)
names(RWMC_MSE) = c("phi_MSE_RWMC", "sigma_MSE_RWMC")
print(RWMC_MSE)

# Coverage probability
lower_ci = rw_post_lower_quantile
upper_ci = rw_post_upper_quantile
RWMC_coverage_prob = apply(theta_test > lower_ci & theta_test < upper_ci, 1, mean)
names(RWMC_coverage_prob) = c("phi_coverage_RWMC", "sigma_coverage_RWMC")
print(RWMC_coverage_prob)

rw_time2 <- Sys.time()

rw_df <- data.frame(
  phi            = theta_test[1, ],
  sigma          = theta_test[2, ],
  rw_phi_mean    = rw_post_means["rw_phi_mean",   ],
  rw_sigma_mean  = rw_post_means["rw_sigma_mean", ]
)

phi_plot <- ggplot(rw_df, aes(x = phi, y = rw_phi_mean)) +
  geom_point(color = "red") +
  geom_abline(linetype = 2, slope = 1, intercept = 0) +
  labs(title = expression(phi), 
       subtitle = paste0("Number of illegal estimates: ", sum(rw_df$rw_phi_mean > 0.8)),
       x = "True φ", y = "RW-Metropolis φ̄") +
  ylim(0, 0.8) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

sigma_plot <- ggplot(rw_df, aes(x = sigma, y = rw_sigma_mean)) +
  geom_point(color = "red") +
  geom_abline(linetype = 2, slope = 1, intercept = 0) +
  labs(title = expression(sigma), 
       subtitle = paste0("Number of illegal estimates: ", sum(rw_df$rw_sigma_mean > 0.8)),
       x = "True σ", y = "RW-Metropolis σ̄") +
  ylim(0, 0.8) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

# Save figure
png("RWMC.png", width = 1600, height = 800, res = 200)
phi_plot + sigma_plot +
  plot_annotation(
    title    = "Random-Walk MC",
    subtitle = paste0("Time taken: ", format(rw_time2 - rw_time1))
  )
dev.off()

# # HMC ---------------------------------------------------------------------
# 
# stan_code <- "
# data {
#   int<lower=1> N;
#   vector[N] y;
#   matrix[N,N] D;
# }
# parameters {
#   real<lower=0> phi;
#   real<lower=0> sigma;
# }
# model {
#   matrix[N,N] Sigma;
#   for (i in 1:N)
#     for (j in 1:N)
#       Sigma[i,j] = sigma * exp(-D[i,j]/phi) + (i == j) * 1e-6;
#   y ~ multi_normal(rep_vector(0, N), Sigma);
# }
# "
# 
# # Pre‐compile the model once
# stan_model <- stan_model(model_code = stan_code)  
# 
# hmc_time1 = Sys.time()
# 
# hmc_post_means <- matrix(NA, nrow = 2, ncol = n_reps,
#                      dimnames = list(c("hmc_phi_mean", "hmc_sigma_mean")))
# 
# for (i in seq_len(n_reps)) {
#   # 1) Pull simulate data
#   S_i <- list( S_test[[i]] )
#   Z_i <- list( Z_test[[i]] )
#   
#   # 2) Build stan data
#   pts <- S_i[[1]]
#   D   <- as.matrix(dist(pts))
#   y   <- as.numeric(Z_i[[1]])
#   stan_data <- list(N = nrow(D), y = y, D = D)
#   
#   # 3) Fit with short chains (you can tweak iter/warmup/chains)
#   fit_i <- sampling(stan_model,
#                     data   = stan_data,
#                     chains = 2,
#                     cores = 2,
#                     iter   = 1000,
#                     warmup = 250,
#                     control = list(
#                       adapt_delta   = 0.7,   # less stringent acceptance
#                       max_treedepth = 8      # shallower trees
#                     ))
#   
#   # 4) Extract and compute posterior means
#   samps <- rstan::extract(fit_i, pars = c("phi", "sigma"), permuted = TRUE)
#   hmc_post_means["hmc_phi_mean", i] <- mean(samps$phi)
#   hmc_post_means["hmc_sigma_mean", i] <- mean(samps$sigma)
# }
# 
# hmc_time2 = Sys.time()
# 
# hmc_df = data.frame(t(rbind(theta_test, hmc_post_means)))
# 
# phi_plot = ggplot(hmc_df) +
#   geom_point(aes(x = phi, y = hmc_phi_mean), col = "red") +
#   geom_abline(linetype = 2, slop = 1, intercept = 0) +
#   labs(title = TeX("\\phi"), x = "Truth", y = "Posterior Mean") +
#   theme_bw() +
#   theme(plot.title = element_text(hjust = 0.5))
# 
# sigma_plot = ggplot(hmc_df) +
#   geom_point(aes(x = sigma, y = hmc_sigma_mean), col = "red") +
#   geom_abline(linetype = 2, slop = 1, intercept = 0) +
#   labs(title = TeX("\\sigma"), x = "Truth", y = "Posterior Mean") +
#   theme_bw() +
#   theme(plot.title = element_text(hjust = 0.5))
#   
# 
# png("HMC.png", width = 1600, height = 800, res = 200)
# phi_plot + sigma_plot + plot_annotation(title = "HMC",
#                                         subtitle = paste0("Time taken: ", format(hmc_time2 - hmc_time1)))
# dev.off()
