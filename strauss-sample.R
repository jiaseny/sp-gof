# Draw samples from a 2D-Strauss process and save to text file.
# Requires the package spatstat:
# https://cran.r-project.org/web/packages/spatstat/

# Parse command line args
args = commandArgs(trailingOnly = TRUE)
n = as.integer(args[1])
l = as.numeric(args[2])
beta0 = as.numeric(args[3])
gamma0 = as.numeric(args[4])
r0 = as.numeric(args[5])
beta = as.numeric(args[6])
gamma = as.numeric(args[7])
r = as.numeric(args[8])
true_dist = as.integer(args[9])  # 0 for p, 1 for q
dist = as.integer(args[10])  # 0 for p, 1 for q
seed = as.integer(args[11])  # Random seed
seed = seed*10 + dist  # Make distinct seeds
res_dir = as.character(args[12])  # Directory for storing results

if (!true_dist || !dist) {  # Use null model params
  beta_q = beta0
  gamma_q = gamma0
  r_q = r0
} else {  # Use alternative model params
  beta_q = beta
  gamma_q = gamma
  r_q = r
}

print(sprintf("n=%d, l=%s, beta0=%.3f, gamma0=%.3f, r0=%.3f, beta=%.3f, gamma=%.3f, r=%.3f, true_dist=%d, job_num=%s, seed=%d, dist=%d, res_dir=%s",
              n, l, beta, gamma, r, beta, gamma, r, true_dist, job_num, seed, dist, res_dir))


suppressMessages(library(spatstat))
set.seed(seed)

sample_strauss <- function(l, beta, gamma, r) {
  # l: size of the square domain
  # beta, gamma, r: parameters of the conditional intensity
  model <- list(cif="strauss",
                par=list(beta=beta, gamma=gamma, r=r),
                w=c(0, l, 0, l))
  res <- rmh(model=model)
  X <- matrix(nrow=res$n, ncol=2)  # Record (x, y) coordinates
  X[,1] <- res$x
  X[,2] <- res$y
  return(X)
}

# Actually draws samples and save to text file
fname = sprintf("strauss-samples-n%d-l%.1f-beta%.3f-gamma%.3f-r%.3f-seed%d.txt",
                n, l, beta, gamma, r, seed)  # seed includes both seed and dist
filename = paste(res_dir, fname, sep="")  # Full filename including path

for (i in 1:n) {
  if (i %% 100 == 1) print(sprintf("Sampled %d processes ...", i))
  X <- sample_strauss(l, beta_q, gamma_q, r_q)
  X <- cbind(rep(i, nrow(X)), X)
  if (i == 1) {
    write.table(X, filename, row.names=FALSE, col.names=c("Index", "x", "y"))
  } else {
    write.table(X, filename, append=TRUE, row.names=FALSE, col.names=FALSE)
  }
}
