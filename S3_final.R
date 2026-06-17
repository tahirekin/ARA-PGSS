# =============================================================================
# ARA-PGSS: Section 5.3 — Defender's Decision Model (Algorithm 3)
#
# Produces Figures 9, 10, and 11 from the paper:
#   Figure 9   Attacker reaction function p(delta | d) and deterrence decomposition
#   Figure 10  Defender loss curve L^D(d) = E[OKL | d] + kappa*d
#   Figure 11  Predictive PMF overlay and defender loss surface
#
# Prerequisites: source S1_S2_final.R first in a clean R session.
# =============================================================================

required_objs <- c("y", "a_clean", "b_clean", "gamma_nom", "n_train",
                   "archetypes", "cost_base", "delta_vals", "delta_vals_nonneg",
                   "t_eval", "a_prev", "b_prev", "a_cl_base", "b_cl_base",
                   "r_cl_base", "p_cl_base", "gamma_vec", "gamma_probs", "Kmax_val",
                   "a011", "a111", "a211", "c1", "kl_gamma", "kl_negbin",
                   "mu_fun", "theme_ara")
missing_objs <- required_objs[!sapply(required_objs, exists)]
if (length(missing_objs) > 0)
  stop("Missing objects from S1_S2_final.R:\n  ", paste(missing_objs, collapse = ", "),
       "\nPlease source S1_S2_final.R first in a clean session.")
stopifnot("cost_base is NA or invalid" = !is.na(cost_base) && cost_base > 0)

library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

# Local parameter overrides for Section 5.3.
# s2 = -1.5 produces stronger deterrence so that E[delta* | d] and P(success | d)
# both vary visibly across d in Figure 9. c_theta = 6 increases stochasticity
# in the theta draw. The attacker cost is reduced to 0.0035 to keep the
# attacker near the deterrence threshold at the FM archetype operating point.
a211 <- -1.5
c1   <- 6

ar_primary <- archetypes$FM
d_grid     <- 0:10
kappa_base <- 0.5
n_samp     <- 500
Kmax_def   <- Kmax_val
cost_def   <- 0.0035
t_eval_def <- t_eval

# =============================================================================
# ALGORITHM 3 IMPLEMENTATION (one-step, H = 1)
#
# For each Monte Carlo replicate:
#   1. Draw gamma ~ Uniform{gamma_vec}
#   2. Run attacker's one-step gray-box optimization over delta_vals
#   3. Draw theta ~ Beta(c_theta * mu, c_theta * (1 - mu))
#   4. Realize attack; advance PGSS filter
#   5. Compute oracle KL loss between attacked and clean NegBin predictives
# =============================================================================

oracle_kl_fun <- function(r_att, p_att, r_cl, p_cl, Kmax = Kmax_def) {
  kl_negbin(r_att, p_att, r_cl, p_cl, Kmax)
}

defender_one_step <- function(
    d_val, t, y, a_prev, b_prev,
    ar           = archetypes$FM,
    gvec         = c(0.7, 0.8, 0.9),
    gprobs       = rep(1/3, 3),
    delta_vals   = delta_vals_nonneg,
    cost_rate    = cost_def,
    Kmax         = Kmax_def,
    n_samples    = n_samp,
    a0_logit     = a011,
    a1_logit     = a111,
    a2_logit     = a211,
    c_theta      = c1) {

  gprobs <- gprobs / sum(gprobs)
  out    <- vector("list", n_samples)

  for (s in seq_len(n_samples)) {
    gam_s  <- sample(gvec, 1, prob = gprobs)
    a_cl_s <- gam_s * a_prev + y[t]; b_cl_s <- gam_s * b_prev + 1
    r_cl_s <- gam_s * a_cl_s;        p_cl_s <- (gam_s * b_cl_s) / (gam_s * b_cl_s + 1)

    klF_raw <- numeric(length(delta_vals))
    klP_raw <- numeric(length(delta_vals))

    for (j in seq_along(delta_vals)) {
      del_j   <- delta_vals[j]
      y_til_j <- max(0L, y[t] + del_j)
      a_j     <- gam_s * a_prev + y_til_j; b_j <- gam_s * b_prev + 1
      r_j     <- gam_s * a_j;              p_j <- (gam_s * b_j) / (gam_s * b_j + 1)
      klF_raw[j] <- as.numeric(kl_gamma(a_j, b_j, a_cl_s, b_cl_s))
      klP_raw[j] <- as.numeric(kl_negbin(r_j, p_j, r_cl_s, p_cl_s, Kmax))
    }

    mu_vec    <- plogis(a0_logit + a1_logit * delta_vals + a2_logit * d_val)
    cost_vec  <- cost_rate * delta_vals^2
    utility   <- (ar$wF * klF_raw + ar$wP * klP_raw) * mu_vec - cost_vec
    delta_opt <- delta_vals[which.max(utility)]

    mu_opt  <- plogis(a0_logit + a1_logit * delta_opt + a2_logit * d_val)
    alpha_s <- c_theta * mu_opt; beta_s <- c_theta * (1 - mu_opt)
    theta_s <- rbeta(1, alpha_s, beta_s)
    succ_s  <- rbinom(1, 1, theta_s)
    y_tilde <- if (succ_s == 1L) max(0L, y[t] + delta_opt) else y[t]

    a_att_s <- gam_s * a_prev + y_tilde; b_att_s <- gam_s * b_prev + 1
    r_att_s <- gam_s * a_att_s;          p_att_s <- (gam_s * b_att_s) / (gam_s * b_att_s + 1)
    okl_s   <- oracle_kl_fun(r_att_s, p_att_s, r_cl_s, p_cl_s)

    out[[s]] <- data.frame(d = d_val, s = s, gamma_draw = gam_s, delta_star = delta_opt,
                           theta_draw = theta_s, success = succ_s, y_tilde = y_tilde,
                           a_att = a_att_s, b_att = b_att_s, r_pred = r_att_s, p_pred = p_att_s,
                           r_cl = r_cl_s, p_cl = p_cl_s, okl = okl_s)
  }
  dplyr::bind_rows(out)
}

# Run Algorithm 3 for all d in d_grid
cat("Running Algorithm 3 for all d in d_grid ...\n")
set.seed(42)
all_d_sims <- lapply(d_grid, function(dval) {
  cat(sprintf("  d = %2d ...", dval))
  sim <- defender_one_step(d_val = dval, t = t_eval_def, y = y,
                           a_prev = a_prev, b_prev = b_prev,
                           ar = ar_primary, gvec = gamma_vec, gprobs = gamma_probs,
                           delta_vals = delta_vals_nonneg, cost_rate = cost_def, n_samples = n_samp)
  cat(sprintf("  E[delta*]=%5.2f  P(succ)=%.3f  E[OKL]=%.4f\n",
              mean(sim$delta_star), mean(sim$success), mean(sim$okl)))
  sim
})
names(all_d_sims) <- as.character(d_grid)

# =============================================================================
# FIGURE 9: ATTACKER REACTION FUNCTION p(delta | d) AND DETERRENCE DECOMPOSITION
#
# Panel A: E[delta* | d] and P(attack succeeds | d) across defense levels.
# Panel B: Distribution of optimal attack magnitudes for d in {0, 5, 10}.
# =============================================================================
cat("Producing Figure 9 ...\n")

prob_ad_df <- do.call(rbind, lapply(d_grid, function(dval) {
  s   <- all_d_sims[[as.character(dval)]]
  data.frame(d = dval, p_no_att = mean(s$delta_star == 0),
             p_att_fail = mean(s$delta_star > 0 & s$success == 0),
             p_att_succ = mean(s$delta_star > 0 & s$success == 1),
             e_delta = mean(s$delta_star), p_success = mean(s$success),
             stringsAsFactors = FALSE)
}))

e_range   <- range(prob_ad_df$e_delta)
p_range   <- range(prob_ad_df$p_success)
scale_p   <- function(p) e_range[1] + (p - p_range[1]) / (diff(p_range) + 1e-10) * diff(e_range)
unscale_p <- function(x) p_range[1] + (x - e_range[1]) / (diff(e_range) + 1e-10) * diff(p_range)
prob_ad_df$p_success_sc <- scale_p(prob_ad_df$p_success)

pD1_B <- ggplot(prob_ad_df, aes(x = d)) +
  geom_line(aes(y = e_delta,       colour = "E[delta* | d]"),  linewidth = 1.0) +
  geom_point(aes(y = e_delta,      colour = "E[delta* | d]"),  size = 2.0) +
  geom_line(aes(y = p_success_sc,  colour = "P(success | d)"), linewidth = 1.0, linetype = "dashed") +
  geom_point(aes(y = p_success_sc, colour = "P(success | d)"), size = 2.0) +
  scale_colour_manual(values = c("E[delta* | d]" = "#5B5EA6", "P(success | d)" = "#C0392B"), name = NULL) +
  scale_x_continuous(breaks = d_grid) +
  scale_y_continuous(
    name     = "E[delta* | d]  (unconditional)",
    sec.axis = sec_axis(transform = ~ unscale_p(.), name = "P(attack succeeds | d)",
                        labels = scales::percent_format(accuracy = 1))) +
  labs(x = "Defense level d") + theme_ara +
  theme(legend.position = "right",
        axis.title.y.right = element_text(colour = "#C0392B"),
        axis.text.y.right  = element_text(colour = "#C0392B"))

d_show_C  <- c(0, 5, 10)
delta_dist_df <- do.call(rbind, lapply(d_show_C, function(dval) {
  data.frame(d = dval, d_label = paste0("d = ", dval),
             delta = all_d_sims[[as.character(dval)]]$delta_star, stringsAsFactors = FALSE)
}))
delta_dist_df$d_label <- factor(delta_dist_df$d_label, levels = paste0("d = ", d_show_C))

edelta_C <- data.frame(
  d_label = factor(paste0("d = ", d_show_C), levels = paste0("d = ", d_show_C)),
  e_delta = sapply(d_show_C, function(dv) mean(all_d_sims[[as.character(dv)]]$delta_star)))

pD1_C <- ggplot(delta_dist_df, aes(x = delta)) +
  geom_histogram(binwidth = 1, fill = "#5B5EA6", colour = "white", alpha = 0.85) +
  geom_vline(data = edelta_C, aes(xintercept = e_delta),
             linetype = "dashed", colour = "#C0392B", linewidth = 0.8) +
  facet_wrap(~ d_label, nrow = 1, scales = "free_y") +
  scale_x_continuous(breaks = seq(0, max(delta_vals_nonneg), 5)) +
  labs(x = "Optimal attack size (delta*)", y = "Count") + theme_ara +
  theme(panel.spacing = unit(1, "lines"))

fig_D1 <- pD1_B / pD1_C +
  plot_annotation(
    title      = "Attacker Reaction Function: p(delta | d) and Deterrence Decomposition",
    tag_levels = "A",
    theme = theme(plot.title = element_text(face = "bold", size = 11),
                  plot.tag   = element_text(face = "bold", size = 10)))

print(fig_D1)
ggsave("Fig_D1_reaction_function.pdf", fig_D1, width = 11, height = 8)

# =============================================================================
# FIGURE 10: DEFENDER LOSS CURVE (Figure 10)
#
# L^D(d; kappa) = E[OKL | d] + kappa * d for kappa in {0.1, 0.5, 1.0}.
# Stacked area: red = expected forecast damage; blue = defense cost.
# =============================================================================
cat("Producing Figure 10 ...\n")

kappa_vals <- c(0.1, 0.5, 1.0)
okl_by_d   <- sapply(d_grid, function(dv) mean(all_d_sims[[as.character(dv)]]$okl))

loss_rows <- list()
for (kap in kappa_vals) {
  for (j in seq_along(d_grid)) {
    dv <- d_grid[j]
    loss_rows[[length(loss_rows) + 1]] <- data.frame(
      kappa = kap, d = dv, okl_mean = okl_by_d[j], def_cost = kap * dv,
      total_loss = okl_by_d[j] + kap * dv, kappa_lbl = paste0("kappa = ", kap))
  }
}
loss_df <- dplyr::bind_rows(loss_rows)

dstar_df <- loss_df %>% dplyr::group_by(kappa) %>%
  dplyr::slice_min(total_loss, n = 1) %>% dplyr::ungroup()

cat("  Optimal defense decisions:\n")
for (i in seq_len(nrow(dstar_df)))
  cat(sprintf("    kappa = %.1f  ->  d* = %d  (total loss = %.4f)\n",
              dstar_df$kappa[i], dstar_df$d[i], dstar_df$total_loss[i]))

loss_long <- loss_df %>%
  tidyr::pivot_longer(cols = c(okl_mean, def_cost), names_to = "component", values_to = "loss") %>%
  dplyr::mutate(component = dplyr::recode(component,
                                          okl_mean = "Attack damage (E[oracle KL])",
                                          def_cost = "Defense cost (kappa x d)"),
                kappa_lbl = paste0("kappa = ", kappa))

kappa_lvls <- paste0("kappa = ", kappa_vals)
loss_df$kappa_lbl   <- factor(loss_df$kappa_lbl,   levels = kappa_lvls)
dstar_df$kappa_lbl  <- factor(dstar_df$kappa_lbl,  levels = kappa_lvls)
loss_long$kappa_lbl <- factor(loss_long$kappa_lbl, levels = kappa_lvls)

fig_D2 <- ggplot() +
  geom_area(data = loss_long, aes(x = d, y = loss, fill = component),
            position = "stack", alpha = 0.65) +
  geom_line(data = loss_df, aes(x = d, y = total_loss), colour = "grey20", linewidth = 1.0) +
  geom_vline(data = dstar_df, aes(xintercept = d), linetype = "dotted", colour = "black", linewidth = 0.8) +
  geom_text(data = dstar_df, aes(x = d + 0.3, y = Inf, label = paste0("d* = ", d)),
            vjust = 1.5, hjust = 0, size = 3.2, colour = "black") +
  scale_fill_manual(values = c("Attack damage (E[oracle KL])" = "#E41A1C",
                               "Defense cost (kappa x d)"     = "#377EB8"),
                    labels = c("Attack damage (E[oracle KL])" = "Attack damage",
                               "Defense cost (kappa x d)"     = "Defense cost")) +
  scale_x_continuous(breaks = seq(0, 10, 2)) +
  facet_wrap(~ kappa_lbl, nrow = 1, scales = "fixed") +
  labs(title = "Defender Loss Curve: Expected Forecast Damage + Defense Cost",
       x = "Defense level d", y = "Loss", fill = NULL) +
  theme_ara + theme(legend.position = "bottom")

print(fig_D2)
ggsave("Fig_D2_defender_loss.pdf", fig_D2, width = 12, height = 5)

# =============================================================================
# FIGURE 11: PREDICTIVE PMF OVERLAY AND DEFENDER LOSS SURFACE (Figure 11)
#
# Panel A: one-step-ahead NegBin PMF under four scenarios (clean, attacked at
# d=0 delta=8, attacked at d=5 delta=4, and defended at d*).
# Panel B: total loss surface L(d; kappa) for kappa in {0.02, 0.10, 0.40}.
# =============================================================================
cat("Producing Figure 11 ...\n")

delta_d0    <- 8L
delta_d5    <- 4L
delta_dstar <- 2L
d_star_D3   <- 10L

a_cl_D3 <- gamma_nom * a_prev + y[t_eval_def]; b_cl_D3 <- gamma_nom * b_prev + 1
r_cl_D3 <- gamma_nom * a_cl_D3; p_cl_D3 <- (gamma_nom * b_cl_D3) / (gamma_nom * b_cl_D3 + 1)

attacked_pred <- function(delta) {
  y_til <- max(0L, y[t_eval_def] + delta)
  a_att <- gamma_nom * a_prev + y_til; b_att <- gamma_nom * b_prev + 1
  r_att <- gamma_nom * a_att;          p_att <- (gamma_nom * b_att) / (gamma_nom * b_att + 1)
  list(r = r_att, p = p_att)
}

pred_d0    <- attacked_pred(delta_d0)
pred_d5    <- attacked_pred(delta_d5)
pred_dstar <- attacked_pred(delta_dstar)

lbl_clean  <- "Clean (no attack)"
lbl_d0     <- sprintf("Attacked (d = 0, delta = %d)", delta_d0)
lbl_d5     <- sprintf("Attacked (d = 5, delta = %d)", delta_d5)
lbl_dstar  <- sprintf("Defended (d*, delta = %d)",    delta_dstar)
scen_lvls  <- c(lbl_clean, lbl_d0, lbl_d5, lbl_dstar)
scen_cols  <- setNames(c("#888888", "#E41A1C", "#4DAF4A", "#377EB8"), scen_lvls)

r_vals   <- c(r_cl_D3, pred_d0$r, pred_d5$r, pred_dstar$r)
p_vals   <- c(p_cl_D3, pred_d0$p, pred_d5$p, pred_dstar$p)
y_max_D3 <- max(sapply(seq_along(r_vals), function(i)
  qnbinom(0.995, size = r_vals[i], prob = 1 - p_vals[i]))) + 5
k_D3     <- 0:y_max_D3

pmf_D3 <- data.frame(
  k    = rep(k_D3, 4),
  prob = c(dnbinom(k_D3, size = r_cl_D3,      prob = 1 - p_cl_D3),
           dnbinom(k_D3, size = pred_d0$r,    prob = 1 - pred_d0$p),
           dnbinom(k_D3, size = pred_d5$r,    prob = 1 - pred_d5$p),
           dnbinom(k_D3, size = pred_dstar$r, prob = 1 - pred_dstar$p)),
  scenario = factor(rep(scen_lvls, each = length(k_D3)), levels = scen_lvls))

pD3_A <- ggplot(pmf_D3, aes(x = k, y = prob, colour = scenario)) +
  geom_line(linewidth = 1.0) + geom_point(size = 0.7, alpha = 0.6) +
  scale_colour_manual(values = scen_cols, name = NULL) +
  labs(x = expression(Y[t+1]), y = "Predictive probability") +
  theme_ara + theme(legend.position = "right")

kappa_D3_vals <- c(0.02, 0.10, 0.40)
loss_D3_df <- do.call(rbind, lapply(kappa_D3_vals, function(kap) {
  total   <- okl_by_d + kap * d_grid
  dstar_k <- d_grid[which.min(total)]
  data.frame(d = d_grid, total_loss = total, kappa = kap,
             kappa_lbl = sprintf("%.2f", kap), d_star = dstar_k)
}))
loss_D3_df$kappa_lbl <- factor(loss_D3_df$kappa_lbl, levels = sprintf("%.2f", kappa_D3_vals))
dstar_D3_df <- loss_D3_df[loss_D3_df$d == loss_D3_df$d_star, ]
dstar_D3_df <- dstar_D3_df[!duplicated(dstar_D3_df$kappa), ]

kappa_cols_D3 <- setNames(c("#E41A1C", "#FF7F00", "#377EB8"), sprintf("%.2f", kappa_D3_vals))

pD3_B <- ggplot(loss_D3_df, aes(x = d, y = total_loss, colour = kappa_lbl, group = kappa_lbl)) +
  geom_line(linewidth = 1.1) + geom_point(size = 1.5) +
  geom_point(data = dstar_D3_df, aes(x = d_star, y = total_loss),
             shape = 18, size = 4, show.legend = FALSE) +
  geom_text(data = dstar_D3_df, aes(x = d_star, y = total_loss, label = paste0("d*=", d_star)),
            vjust = -0.8, size = 3.0, fontface = "bold", show.legend = FALSE) +
  scale_colour_manual(values = kappa_cols_D3, name = "kappa") +
  scale_x_continuous(breaks = d_grid) +
  labs(x = "Defense level d", y = "Total loss") + theme_ara + theme(legend.position = "right")

fig_D3 <- pD3_A / pD3_B +
  plot_annotation(title = "Predictive PMF and Defender Loss Surface",
                  tag_levels = "A",
                  theme = theme(plot.title = element_text(face = "bold", size = 11),
                                plot.tag   = element_text(face = "bold", size = 10)))

print(fig_D3)
ggsave("Fig_D3_pmf_overlay.pdf", fig_D3, width = 10, height = 8)

cat("\nAll Section 5.3 figures saved:\n")
cat("  Fig_D1_reaction_function.pdf\n")
cat("  Fig_D2_defender_loss.pdf\n")
cat("  Fig_D3_pmf_overlay.pdf\n")
cat(sprintf("Global a211 at end of script: %.1f\n", a211))
