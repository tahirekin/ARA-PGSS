# =============================================================================
# ARA-PGSS: Section 6 — Real-World Illustration with Avazu Click-Stream Data
#
# Produces four figures:
#   Figure 12  (S6_A_filter_fit.pdf)
#              Hourly click counts with PGSS posterior mean and one-step forecast.
#   Figure 13  (S6_B_attack_trajectory.pdf)
#              Filter trajectories and cumulative forecast inflation under three
#              fraud scenarios over a 48-hour window.
#   Figure 14a (S6_C_predictive_pmf.pdf)
#              One-step-ahead predictive distributions: clean, attacked, defended.
#   Figure 14b (S6_D_defender_loss.pdf)
#              Defender total-loss surface L(d; kappa) = E[KL | d] + kappa*d.
#
# Prerequisites: source S1_S2_final.R first in a clean R session.
# The Avazu data file "avazu.xlsx" must be in the working directory.
# =============================================================================

required_objs <- c("kl_gamma", "kl_negbin", "a011", "a111", "a211", "c1",
                   "gamma_vec", "gamma_probs", "Kmax_val", "theme_ara")
missing_req <- required_objs[!sapply(required_objs, exists)]
if (length(missing_req) > 0)
  stop("Missing from S1_S2_final.R:\n  ", paste(missing_req, collapse = ", "),
       "\nPlease source S1_S2_final.R first.")

library(readxl); library(ggplot2); library(dplyr)
library(tidyr);  library(patchwork)

# Local parameter overrides matching Section 5.3 (S3_final.R convention).
# a211_av and c1_av are used only within this script; the global a211 is unchanged.
a211_av     <- -1.5
c1_av       <-  6
cost_def_av <- 0.0035

# =============================================================================
# SECTION 6.1: LOAD AVAZU DATA AND FIT PGSS (Figure 12)
# =============================================================================
cat("=== 6.1  Loading Avazu data and fitting PGSS ===\n")

av          <- read_excel("avazu.xlsx")
colnames(av)<- c("hour_raw", "site_domain", "count")

# Five overnight hours absent from the raw data are imputed as zero,
# consistent with the high empirical zero-rate during overnight periods
# and with the PGSS zero-observation update (a_t = gamma * a_{t-1}).
missing_rows <- data.frame(
  hour_raw    = c(14102421L, 14102422L, 14102921L, 14103000L, 14103022L),
  site_domain = "6a9dcbd8", count = 0L, stringsAsFactors = FALSE)
av <- bind_rows(av, missing_rows) %>%
  arrange(hour_raw) %>%
  mutate(t = seq_len(n()), hs_ = as.character(hour_raw),
         hour_of_day = as.integer(substr(hs_, 7, 8)),
         day = as.integer(substr(hs_, 5, 6))) %>%
  select(-hs_)

n_av <- nrow(av)
y_av <- av$count
cat(sprintf("  %d hours after zero-imputation | mean=%.2f | max=%d\n",
            n_av, mean(y_av), max(y_av)))

# Fit gamma by maximizing the marginal log-likelihood of the one-step-ahead
# NegBin predictive distribution on the first 168 hours (one full week).
n_train_av <- 168L
a0_av <- 1.0; b0_av <- 1.0

log_ml_fun <- function(gam) {
  a_t <- gam * a0_av + y_av[1L]; b_t <- gam * b0_av + 1; ll <- 0
  for (tt in 2L:n_train_av) {
    r_p <- gam * a_t; p_p <- (gam * b_t) / (gam * b_t + 1)
    ll  <- ll + dnbinom(y_av[tt], size = r_p, prob = p_p, log = TRUE)
    a_t <- gam * a_t + y_av[tt]; b_t <- gam * b_t + 1
  }
  ll
}

gam_grid <- seq(0.70, 0.97, by = 0.01)
ml_vals  <- sapply(gam_grid, log_ml_fun)
gam_av   <- gam_grid[which.max(ml_vals)]
cat(sprintf("  Fitted gamma (MLE, first 168 hrs): %.2f\n", gam_av))

# PGSS filter over all 240 hours
a_av <- numeric(n_av); b_av <- numeric(n_av)
lam_av <- numeric(n_av); fcast_m <- numeric(n_av)

a_av[1L]    <- gam_av * a0_av + y_av[1L]; b_av[1L] <- gam_av * b0_av + 1
lam_av[1L]  <- a_av[1L] / b_av[1L]
fcast_m[1L] <- a0_av / b0_av

for (tt in 2L:n_av) {
  fcast_m[tt] <- a_av[tt - 1L] / b_av[tt - 1L]
  a_av[tt]    <- gam_av * a_av[tt - 1L] + y_av[tt]
  b_av[tt]    <- gam_av * b_av[tt - 1L] + 1
  lam_av[tt]  <- a_av[tt] / b_av[tt]
}

# Figure 12
fa_df <- data.frame(t = 1L:n_av, y_obs = y_av, lam_post = lam_av, fcast_mean = fcast_m)
day_starts <- seq(4L, n_av, by = 24L)
day_peaks  <- data.frame(xmin = day_starts + 0.5, xmax = pmin(day_starts + 12L, n_av) + 0.5)

fig_A <- ggplot(fa_df, aes(x = t)) +
  geom_rect(data = day_peaks, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "#E8F4FD", alpha = 0.7) +
  geom_col(aes(y = y_obs), fill = "grey72", color = NA, width = 0.85) +
  geom_line(aes(y = lam_post,    color = "Posterior mean"),    linewidth = 0.9) +
  geom_line(aes(y = fcast_mean,  color = "One-step forecast"), linewidth = 0.75, linetype = "dashed") +
  scale_color_manual(values = c("Posterior mean" = "#1B7837", "One-step forecast" = "#D6604D"), name = NULL) +
  scale_x_continuous(breaks = c(1L, 25L, 49L, 73L, 97L, 121L, 145L, 169L, 193L, 217L),
                     labels = paste0("Day ", 1:10)) +
  labs(title = "Avazu Hourly Click Counts and PGSS Filter",
       x = "Hour index (t)", y = expression(Y[t])) +
  theme_ara + theme(legend.position = "bottom", axis.text.x = element_text(angle = 30, hjust = 1))

print(fig_A)
ggsave("S6_A_filter_fit.pdf", fig_A, width = 11, height = 4.5)

# =============================================================================
# SECTION 6.2: SEQUENTIAL CLICK-FRAUD INJECTION ATTACKS (Figure 13)
#
# Three FM-motivated fraud scenarios over a 48-hour window (t = 103..150):
#   Low-intensity: delta = 2 every hour regardless of traffic level.
#   Adaptive:      delta = 5 whenever the one-step-ahead forecast >= 5.
#   Aggressive:    delta = 8 for the first 24 hours, then stops.
# =============================================================================
cat("\n=== 6.2  Sequential click-fraud injection ===\n")

t_start_av  <- 103L
H_av        <- 48L
t_window_av <- t_start_av:(t_start_av + H_av - 1L)

run_avazu_attack <- function(y_full, t_start, H, gam, a_init, b_init, delta_rule, label) {
  lam_seq <- numeric(H); fcast_seq <- numeric(H)
  delta_seq <- numeric(H); y_til_seq <- numeric(H)
  a_p <- a_init; b_p <- b_init

  for (k in seq_len(H)) {
    tau   <- t_start + k - 1L; y_t <- y_full[tau]
    fm_prev <- gam * a_p
    del_k   <- delta_rule(k, fm_prev)
    y_til   <- max(0L, y_t + del_k)
    a_p <- gam * a_p + y_til; b_p <- gam * b_p + 1
    lam_seq[k]   <- a_p / b_p
    fcast_seq[k] <- gam * a_p
    delta_seq[k] <- del_k; y_til_seq[k] <- y_til
  }
  data.frame(k = seq_len(H), t = t_start + seq_len(H) - 1L,
             y_obs = y_full[t_window_av], delta = delta_seq,
             lam_att = lam_seq, fcast_att = fcast_seq, scenario = label)
}

a_win_init <- a_av[t_start_av - 1L]; b_win_init <- b_av[t_start_av - 1L]

att_clean <- run_avazu_attack(y_av, t_start_av, H_av, gam_av, a_win_init, b_win_init,
                              function(k, fm) 0L, "Clean (no attack)")
att_low   <- run_avazu_attack(y_av, t_start_av, H_av, gam_av, a_win_init, b_win_init,
                              function(k, fm) 2L, "Low-intensity (delta=2)")
att_adapt <- run_avazu_attack(y_av, t_start_av, H_av, gam_av, a_win_init, b_win_init,
                              function(k, fm) if (fm >= 5) 5L else 0L,
                              "Adaptive (delta=5, E[Y]>=5)")
att_aggr  <- run_avazu_attack(y_av, t_start_av, H_av, gam_av, a_win_init, b_win_init,
                              function(k, fm) if (k <= 24L) 8L else 0L,
                              "Aggressive (delta=8, 24-hr burst)")

traj_all <- bind_rows(att_clean, att_low, att_adapt, att_aggr)
traj_all$scenario <- factor(traj_all$scenario, levels = c(
  "Clean (no attack)", "Low-intensity (delta=2)",
  "Adaptive (delta=5, E[Y]>=5)", "Aggressive (delta=8, 24-hr burst)"))

scen_colors <- c("Clean (no attack)"              = "grey40",
                 "Low-intensity (delta=2)"         = "#4DAF4A",
                 "Adaptive (delta=5, E[Y]>=5)"    = "#377EB8",
                 "Aggressive (delta=8, 24-hr burst)" = "#E41A1C")

pB1 <- ggplot(traj_all, aes(x = t, y = lam_att, color = scenario)) +
  annotate("rect", xmin = 143.5, xmax = 159.5, ymin = -Inf, ymax = Inf,
           fill = "#FEE8C8", alpha = 0.45) +
  geom_line(linewidth = 1.0) +
  geom_point(data = att_clean, aes(x = t, y = y_obs), inherit.aes = FALSE,
             color = "grey50", size = 1.2, shape = 16, alpha = 0.75) +
  scale_color_manual(values = scen_colors, name = NULL) +
  scale_x_continuous(breaks = seq(t_start_av, t_start_av + H_av, by = 12L),
                     labels = paste0("t = ", seq(t_start_av, t_start_av + H_av, by = 12L))) +
  annotate("text", x = 151.5, y = max(traj_all$lam_att) * 0.97,
           label = "Surge", size = 2.6, hjust = 0.5, color = "grey30") +
  labs(y = expression(E(lambda[t] ~ "|" ~ D[t])), x = "Hour index (t)") +
  theme_ara + theme(legend.position = "bottom", axis.text.x = element_text(angle = 30, hjust = 1))

clean_fcast <- att_clean$fcast_att
cum_inf_df <- traj_all %>%
  group_by(scenario) %>%
  mutate(fcast_clean = clean_fcast, inflation = fcast_att - fcast_clean,
         cum_inf = cumsum(pmax(0, inflation))) %>% ungroup()

pB2 <- ggplot(cum_inf_df, aes(x = t, y = cum_inf, color = scenario)) +
  geom_line(linewidth = 1.0) +
  scale_color_manual(values = scen_colors, name = NULL) +
  scale_x_continuous(breaks = seq(t_start_av, t_start_av + H_av, by = 12L),
                     labels = paste0("t = ", seq(t_start_av, t_start_av + H_av, by = 12L))) +
  labs(y = "Cumulative forecast inflation (clicks)", x = "Hour index (t)") +
  theme_ara + theme(legend.position = "bottom", axis.text.x = element_text(angle = 30, hjust = 1))

fig_B <- pB1 / pB2 +
  plot_annotation(title = "Sequential Click-Fraud Attack: Filter Trajectory and Forecast Inflation")

print(fig_B)
ggsave("S6_B_attack_trajectory.pdf", fig_B, width = 11, height = 8)

CPC <- 2.0
cum_summary <- cum_inf_df %>%
  group_by(scenario) %>%
  summarise(total_injected_clicks = round(sum(delta), 1),
            cumulative_inflation  = round(sum(pmax(0, inflation)), 1),
            extra_spend_USD       = round(sum(pmax(0, inflation)) * CPC, 2),
            .groups = "drop")
cat("\n  48-hour cumulative impact (CPC = $2.00):\n")
print(cum_summary)

# =============================================================================
# SECTION 6.3: EVALUATION PERIOD AND ATTACKER OPTIMAL delta*
# =============================================================================
cat("\n=== 6.3  Evaluation at t = 150 ===\n")

t_eval_av  <- 150L
y_eval_av  <- y_av[t_eval_av]
a_prev_av  <- a_av[t_eval_av - 1L]
b_prev_av  <- b_av[t_eval_av - 1L]

a_cl_av <- gam_av * a_prev_av + y_eval_av; b_cl_av <- gam_av * b_prev_av + 1
r_cl_av <- gam_av * a_cl_av;              p_cl_av <- (gam_av * b_cl_av) / (gam_av * b_cl_av + 1)
fcast_clean_av <- a_cl_av / b_cl_av

cat(sprintf("  t=%d | y=%d | clean forecast = %.2f clicks\n", t_eval_av, y_eval_av, fcast_clean_av))

delta_vals_av <- 0L:20L

klP_vec_av <- function(delta, gam, a_p, b_p, y_t, Kmax = Kmax_val) {
  y_til <- max(0L, y_t + delta)
  a_j   <- gam * a_p + y_til; b_j <- gam * b_p + 1
  r_j   <- gam * a_j;         p_j <- (gam * b_j) / (gam * b_j + 1)
  a_cl  <- gam * a_p + y_t;   b_cl <- gam * b_p + 1
  r_cl  <- gam * a_cl;        p_cl <- (gam * b_cl) / (gam * b_cl + 1)
  list(klP = kl_negbin(r_j, p_j, r_cl, p_cl, Kmax),
       r_att = r_j, p_att = p_j, r_cl = r_cl, p_cl = p_cl,
       fcast_att = a_j / b_j, fcast_cl = a_cl / b_cl)
}

utils_d0 <- sapply(delta_vals_av, function(del) {
  kl   <- klP_vec_av(del, gam_av, a_prev_av, b_prev_av, y_eval_av)
  mu_v <- plogis(a011 + a111 * del + a211_av * 0)
  kl$klP * mu_v - cost_def_av * del^2
})
delta_star_av    <- delta_vals_av[which.max(utils_d0)]
kl_star          <- klP_vec_av(delta_star_av, gam_av, a_prev_av, b_prev_av, y_eval_av)
fcast_attacked_av <- kl_star$fcast_att

cat(sprintf("  FM delta* at d=0: %d  |  attacked forecast: %.2f clicks\n",
            delta_star_av, fcast_attacked_av))
cat(sprintf("  Excess expected spend: $%.2f/hr (CPC=$%.2f)\n",
            (fcast_attacked_av - fcast_clean_av) * CPC, CPC))

# =============================================================================
# SECTION 6.4: DEFENDER ONE-STEP ARA (Algorithm 3 on Avazu parameters)
# =============================================================================
cat("\n=== 6.4  Defender one-step ARA ===\n")

n_samp_av <- 500L
d_grid_av <- 0L:10L
kappa_vals <- c(0.02, 0.10, 0.40)

defender_one_step_av <- function(
    d_val, t_idx, y_full, a_prev, b_prev, gam_nom,
    delta_vals, cost_rate, n_samples = n_samp_av, Kmax = Kmax_val,
    a0 = a011, a1 = a111, a2 = a211_av, c_th = c1_av) {

  out <- vector("list", n_samples)
  y_t <- y_full[t_idx]
  a_cl <- gam_nom * a_prev + y_t; b_cl <- gam_nom * b_prev + 1
  r_cl <- gam_nom * a_cl;         p_cl <- (gam_nom * b_cl) / (gam_nom * b_cl + 1)

  klP_full <- sapply(delta_vals, function(del) {
    y_til <- max(0L, y_t + del)
    a_j   <- gam_nom * a_prev + y_til; b_j <- gam_nom * b_prev + 1
    r_j   <- gam_nom * a_j;            p_j <- (gam_nom * b_j) / (gam_nom * b_j + 1)
    kl_negbin(r_j, p_j, r_cl, p_cl, Kmax)
  })
  mu_vec    <- plogis(a0 + a1 * delta_vals + a2 * d_val)
  cost_vec  <- cost_rate * delta_vals^2
  utility   <- klP_full * mu_vec - cost_vec
  delta_opt <- delta_vals[which.max(utility)]

  y_til_opt <- max(0L, y_t + delta_opt)
  a_att_opt <- gam_nom * a_prev + y_til_opt; b_att_opt <- gam_nom * b_prev + 1
  r_att_opt <- gam_nom * a_att_opt;          p_att_opt <- (gam_nom * b_att_opt) / (gam_nom * b_att_opt + 1)

  mu_opt <- plogis(a0 + a1 * delta_opt + a2 * d_val)
  al_opt <- c_th * mu_opt; be_opt <- c_th * (1 - mu_opt)

  for (s in seq_len(n_samples)) {
    theta_s <- rbeta(1L, max(al_opt, 1e-6), max(be_opt, 1e-6))
    succ_s  <- rbinom(1L, 1L, theta_s)
    y_tilde <- if (succ_s == 1L) y_til_opt else y_t

    a_att_s <- gam_nom * a_prev + y_tilde; b_att_s <- gam_nom * b_prev + 1
    r_att_s <- gam_nom * a_att_s;          p_att_s <- (gam_nom * b_att_s) / (gam_nom * b_att_s + 1)
    okl_s   <- kl_negbin(r_att_s, p_att_s, r_cl, p_cl, Kmax)

    out[[s]] <- data.frame(d = d_val, s = s, delta_star = delta_opt, success = succ_s,
                           y_tilde = y_tilde, r_att = r_att_s, p_att = p_att_s,
                           r_cl = r_cl, p_cl = p_cl, okl = okl_s)
  }
  dplyr::bind_rows(out)
}

set.seed(42)
cat("  Running Algorithm 3 on Avazu parameters (d = 0..10) ...\n")
all_d_av <- lapply(d_grid_av, function(dv) {
  cat(sprintf("    d = %2d ...", dv))
  res <- defender_one_step_av(d_val = dv, t_idx = t_eval_av, y_full = y_av,
                               a_prev = a_prev_av, b_prev = b_prev_av, gam_nom = gam_av,
                               delta_vals = delta_vals_av, cost_rate = cost_def_av)
  cat(sprintf("  delta*=%d  E[KL]=%.4f  P(succ)=%.3f\n",
              res$delta_star[1L], mean(res$okl), mean(res$success)))
  res
})
sim_av <- dplyr::bind_rows(all_d_av)

loss_av <- sim_av %>%
  group_by(d) %>%
  summarise(E_kl = mean(okl), E_delta = mean(delta_star), P_succ = mean(success),
            r_cl_mn = mean(r_cl), p_cl_mn = mean(p_cl), .groups = "drop") %>%
  crossing(kappa = kappa_vals) %>%
  mutate(total_loss = E_kl + kappa * d,
         kappa_label = factor(paste0("kappa = ", kappa),
                              levels = paste0("kappa = ", sort(kappa_vals))))

d_star_av <- loss_av %>%
  group_by(kappa, kappa_label) %>%
  slice_min(total_loss, n = 1L, with_ties = FALSE) %>% ungroup()

cat("\n  Defender optimal d* by kappa:\n")
for (i in seq_len(nrow(d_star_av)))
  cat(sprintf("    kappa=%.2f -> d*=%d  (total loss=%.4f)\n",
              d_star_av$kappa[i], d_star_av$d[i], d_star_av$total_loss[i]))

# =============================================================================
# FIGURE 14 (left panel): ONE-STEP-AHEAD PREDICTIVE PMF (S6_C_predictive_pmf.pdf)
# =============================================================================
cat("\n--- Figure 14 (left): Predictive PMF overlay ---\n")

d_star_base <- d_star_av$d[d_star_av$kappa == 0.10]

r_cl_c   <- kl_star$r_cl;  p_cl_c   <- kl_star$p_cl;  fcast_cl_c <- kl_star$fcast_cl
r_att_c  <- kl_star$r_att; p_att_c  <- kl_star$p_att; fcast_at_c <- kl_star$fcast_att

def_sims <- sim_av %>% filter(d == d_star_base)
r_def    <- mean(def_sims$r_att); p_def <- mean(def_sims$p_att)
fcast_df_c <- r_def * (1 - p_def) / p_def

k_max_c  <- max(40L, qnbinom(0.995, size = r_att_c, prob = p_att_c))
k_grid_c <- 0L:k_max_c

scen_labels_c <- c("Clean (no attack)",
                   sprintf("Attacked  (d = 0, delta = %d)", delta_star_av),
                   sprintf("Defended  (d = %d, kappa = 0.10)", d_star_base))
pmf_df <- data.frame(
  k    = rep(k_grid_c, 3L),
  prob = c(dnbinom(k_grid_c, size = r_cl_c,  prob = p_cl_c),
           dnbinom(k_grid_c, size = r_att_c, prob = p_att_c),
           dnbinom(k_grid_c, size = r_def,   prob = p_def)),
  scenario = rep(scen_labels_c, each = length(k_grid_c)))
pmf_df$scenario <- factor(pmf_df$scenario, levels = scen_labels_c)
pmf_colors <- setNames(c("grey40", "#E41A1C", "#1B7837"), scen_labels_c)

fig_C <- ggplot(pmf_df, aes(x = k, y = prob, color = scenario, group = scenario)) +
  geom_line(linewidth = 1.0) + geom_point(size = 1.2, alpha = 0.8) +
  geom_vline(xintercept = fcast_cl_c, color = "grey40",  linetype = "dashed", linewidth = 0.65) +
  geom_vline(xintercept = fcast_at_c, color = "#E41A1C", linetype = "dashed", linewidth = 0.65) +
  geom_vline(xintercept = fcast_df_c, color = "#1B7837", linetype = "dashed", linewidth = 0.65) +
  annotate("text", x = fcast_cl_c + 0.4, y = max(pmf_df$prob) * 0.90,
           label = sprintf("%.1f", fcast_cl_c), color = "grey30", size = 3.0, hjust = 0) +
  annotate("text", x = fcast_at_c + 0.4, y = max(pmf_df$prob) * 0.80,
           label = sprintf("%.1f", fcast_at_c), color = "#E41A1C", size = 3.0, hjust = 0) +
  annotate("text", x = fcast_df_c + 0.4, y = max(pmf_df$prob) * 0.70,
           label = sprintf("%.1f\n(defended)", fcast_df_c), color = "#1B7837", size = 3.0, hjust = 0) +
  scale_color_manual(values = pmf_colors, name = NULL) +
  labs(title = "One-Step-Ahead Predictive Distributions: Clean, Attacked, and Defended",
       x = expression(Y[t + 1]), y = "Predictive probability") +
  theme_ara + theme(legend.position = "bottom")

print(fig_C)
ggsave("S6_C_predictive_pmf.pdf", fig_C, width = 9.5, height = 5.5)

# =============================================================================
# FIGURE 14 (right panel): DEFENDER LOSS SURFACE (S6_D_defender_loss.pdf)
# =============================================================================
cat("\n--- Figure 14 (right): Defender loss surface ---\n")

kappa_colors <- setNames(c("#1F78B4", "#FF7F00", "#33A02C"),
                         paste0("kappa = ", sort(kappa_vals)))

fig_D <- ggplot(loss_av, aes(x = d, y = total_loss, color = kappa_label, group = kappa_label)) +
  geom_line(linewidth = 1.1) + geom_point(size = 1.8, alpha = 0.7) +
  geom_point(data = d_star_av, aes(x = d, y = total_loss, color = kappa_label),
             shape = 23L, size = 4.5, fill = "white", stroke = 1.5) +
  geom_text(data = d_star_av,
            aes(x = d + 0.35, y = total_loss * 1.07, label = paste0("d* = ", d), color = kappa_label),
            size = 3.0, hjust = 0, show.legend = FALSE) +
  scale_color_manual(values = kappa_colors, name = NULL) +
  scale_x_continuous(breaks = 0L:10L) +
  labs(title = "Defender Loss Surface: L(d; kappa) = E[KL | d] + kappa * d",
       x = "Defense level d", y = expression(L(d, kappa))) +
  theme_ara + theme(legend.position = "bottom")

print(fig_D)
ggsave("S6_D_defender_loss.pdf", fig_D, width = 9.5, height = 5.5)

cat("\nSection 6 complete. Figures saved:\n")
for (f in c("S6_A_filter_fit.pdf", "S6_B_attack_trajectory.pdf",
            "S6_C_predictive_pmf.pdf", "S6_D_defender_loss.pdf"))
  cat(sprintf("  %s\n", f))
