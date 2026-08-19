# Optional base-R illustration of a two-by-two DiD regression
#
# Python tests remain the required path. GitHub Actions also runs this
# script in a separate job (`optional-r`) so the illustration cannot drift
# silently. A local clone without R can still use the Python laboratory.
#
# The regression is the standard representation
#   y ~ treated_group + post + treated_group:post
# on a simulated two-period panel. Parallel trends hold by construction in
# this DGP. The interaction is the ATT in the 2x2 design.
#
# Copyright 2026 Dr. Pavanam Thomas
# License: MIT

set.seed(42)

n_treat <- 80L
n_control <- 80L
att <- 2.0
sigma <- 1.0

n <- n_treat + n_control
treated_group <- c(rep(1L, n_treat), rep(0L, n_control))
a <- rnorm(n, mean = 0, sd = 1)
time_effects <- c("0" = 0.0, "1" = 0.8)

unit <- integer(0)
period <- integer(0)
g <- integer(0)
post <- integer(0)
y <- numeric(0)

for (i in seq_len(n)) {
  for (t in 0:1) {
    e <- rnorm(1L, mean = 0, sd = sigma)
    y0 <- a[i] + time_effects[as.character(t)] + e
    treated_now <- as.integer(treated_group[i] == 1L && t == 1L)
    unit <- c(unit, i)
    period <- c(period, t)
    g <- c(g, treated_group[i])
    post <- c(post, t)
    y <- c(y, y0 + att * treated_now)
  }
}

panel <- data.frame(
  unit = unit,
  period = period,
  treated_group = g,
  post = post,
  y = y
)

fit <- lm(y ~ treated_group + post + treated_group:post, data = panel)
print(summary(fit))

cell_means <- aggregate(y ~ treated_group + post, data = panel, FUN = mean)
print(cell_means)

did_means <- (
  cell_means$y[cell_means$treated_group == 1 & cell_means$post == 1] -
    cell_means$y[cell_means$treated_group == 1 & cell_means$post == 0]
) - (
  cell_means$y[cell_means$treated_group == 0 & cell_means$post == 1] -
    cell_means$y[cell_means$treated_group == 0 & cell_means$post == 0]
)
cat("2x2 DiD of cell means:", did_means, "\n")
cat("Regression interaction:", coef(fit)["treated_group:post"], "\n")
