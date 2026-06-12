# =============================================================================
# ARA-PGSS: Sections 5.1 and 5.2 — One-Step and Sequential Attacker Analysis
#
# Covers:
#   S1.0  Cost sensitivity and calibration (Figure 1)
#   S1.1  Utility profile decomposition across three cost regimes (Figure 2)
#   S1.2  Posterior distribution shifts by archetype (Figure 3)
#   S1.3  Sensitivity heatmap and profitable-attack threshold (Supplemental Figure 1)
#   S1.4  Gray-box vs. white-box utility curves (Figure 4)
#   S1.5  Attack success probability by defense level (Figure 5)
#   S1.6  Sensitivity of attack success to logistic parameters (Supplemental Figure 2)
#   S2.1  Clean vs. attacked PGSS filter trajectory (Figure 6)
#   S2.2  Cumulative KL divergence over horizon (Figure 7)
#   S2.4  Cumulative attacker utility vs. horizon H (Figure 8)
#
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(viridis)

# =============================================================================
# SECTION 0: DATA-GENERATING PROCESS AND PGSS FIT
# =============================================================================

set.seed(7)
n_total  <- 60
y        <- numeric(n_total)
lam      <- numeric(n_total)
gam_sim  <- 0.8
a0_sim   <- 5
b0_sim   <- 1
lam0     <- 1.0

# Multiplicative scaled Beta state evolution: lambda_t = (lambda_{t-1}/gamma)*epsilon_t
# matching equation (3) in the paper.
eps    <- rbeta(1, gam_sim * a0_sim, (1 - gam_sim) * a0_sim)
lam[1] <- (lam0 / gam_sim) * eps
y[1]   <- rpois(1, lam[1])

a_sim    <- numeric(n_total)
b_sim    <- numeric(n_total)
a_sim[1] <- gam_sim * a0_sim + y[1]
b_sim[1] <- gam_sim * b0_sim + 1

for (i in 2:n_total) {
  eps      <- rbeta(1, gam_sim * a_sim[i - 1], (1 - gam_sim) * a_sim[i - 1])
  lam[i]   <- (lam[i - 1] / gam_sim) * eps
  y[i]     <- rpois(1, lam[i])
  a_sim[i] <- gam_sim * a_sim[i - 1] + y[i]
  b_sim[i] <- gam_sim * b_sim[i - 1] + 1
  lam[i]   <- rgamma(1, a_sim[i], b_sim[i])
  y[i]     <- rpois(1, lam[i])
}

# PGSS fit on first 50 observations
n_train   <- 50
gamma_nom <- gam_sim

a_clean <- numeric(n_train)
b_clean <- numeric(n_train)
r_clean <- numeric(n_train)
p_clean <- numeric(n_train)

a0 <- 1; b0 <- 1
a_clean[1] <- gamma_nom * a0 + y[1]
b_clean[1] <- gamma_nom * b0 + 1
r_clean[1] <- gamma_nom * a0
p_clean[1] <- (gamma_nom * b0) / (gamma_nom * b0 + 1)

for (t in 2:n_train) {
  a_clean[t] <- gamma_nom * a_clean[t - 1] + y[t]
  b_clean[t] <- gamma_nom * b_clean[t - 1] + 1
  r_clean[t] <- gamma_nom * a_clean[t - 1]
  p_clean[t] <- (gamma_nom * b_clean[t - 1]) / (gamma_nom * b_clean[t - 1] + 1)
}

# =============================================================================
# SECTION 0B: SHARED HELPERS
# =============================================================================

# KL divergence between Gamma filtering distributions, eq. (14)
kl_gamma <- function(a1, b1, a2, b2) {
  a1 <- as.numeric(a1)[1]; b1 <- as.numeric(b1)[1]
  a2 <- as.numeric(a2)[1]; b2 <- as.numeric(b2)[1]
  val <- a1 * log(b1 / b2) - lgamma(a1) + lgamma(a2) +
    (a1 - a2) * digamma(a1) + a2 * (b1 - b2) / b2
  max(0, as.numeric(val))
}

# KL divergence between NegBin predictive distributions, eq. (15)
kl_negbin <- function(r1, p1, r2, p2, Kmax = 500) {
  r1 <- as.numeric(r1)[1]; p1 <- as.numeric(p1)[1]
  r2 <- as.numeric(r2)[1]; p2 <- as.numeric(p2)[1]
  k  <- 0:Kmax
  Pk <- dnbinom(k, size = r1, prob = 1 - p1) + 1e-12
  Qk <- dnbinom(k, size = r2, prob = 1 - p2) + 1e-12
  max(0, as.numeric(sum(Pk * log(Pk / Qk))))
}

# Logistic attack-success model, eq. (17)
a011 <-  0.0
a111 <-  0.5
a211 <- -1.0
c1   <- 10

alpha_fun <- function(delta, d) { eta <- a011 + a111*delta + a211*d; mu <- 1/(1+exp(-eta)); c1*mu }
beta_fun  <- function(delta, d) { eta <- a011 + a111*delta + a211*d; mu <- 1/(1+exp(-eta)); c1*(1-mu) }
mu_fun    <- function(delta, d) { eta <- a011 + a111*delta + a211*d; 1/(1+exp(-eta)) }

# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================

delta_vals        <- -5:30
delta_vals_nonneg <- 0:30
t_eval      <- n_train + 1
cost_base   <- 0.5
Kmax_val    <- 500
n_samp      <- 1000
gamma_vec   <- c(0.7, 0.8, 0.9)
gamma_probs <- rep(1/3, 3)
d_base      <- 0

# Two attacker archetypes (Section 5, equations (20)-(21))
archetypes <- list(
  SP = list(wF = 1.0, wP = 0.0, label = "SP (wF=1, wP=0)", label_short = "SP", color = "#E41A1C"),
  FM = list(wF = 0.0, wP = 1.0, label = "FM (wF=0, wP=1)", label_short = "FM", color = "#377EB8")
)

a_prev    <- a_clean[n_train]
b_prev    <- b_clean[n_train]
a_cl_base <- gamma_nom * a_prev + y[t_eval]
b_cl_base <- gamma_nom * b_prev + 1
r_cl_base <- gamma_nom * a_cl_base
p_cl_base <- (gamma_nom * b_cl_base) / (gamma_nom * b_cl_base + 1)

theme_ara <- theme_minimal(base_size = 13) +
  theme(plot.title    = element_text(face = "bold", size = 13),
        plot.subtitle = element_text(size = 11, color = "grey40"),
        legend.position = "bottom")

# =============================================================================
# S1.0  COST SENSITIVITY ANALYSIS (Figure 1)
#
# White-box one-step utility with raw KL (no normalization) and quadratic cost
# (Bruckner & Scheffer 2011): U^A(delta) = wF*KL_F + wP*KL_P - c*delta^2.
# Sweeps c to identify the FM profitability threshold c* and the SP threshold.
# =============================================================================
cat("Running S1.0 ...\n")

one_step_wb_raw <- function(t, delta_vals, y,
                            a_clean_prev, b_clean_prev,
                            a_att_prev, b_att_prev,
                            gamma = 0.8, d_fixed = 0,
                            wF = 0.5, wP = 0.5,
                            cost_rate = 0.1, Kmax = 500) {
  cost_rate <- as.numeric(cost_rate)[1]
  wF <- as.numeric(wF)[1]; wP <- as.numeric(wP)[1]; gamma <- as.numeric(gamma)[1]
  
  a_cl <- gamma * a_clean_prev + y[t]; b_cl <- gamma * b_clean_prev + 1
  r_cl <- gamma * a_cl; p_cl <- (gamma * b_cl) / (gamma * b_cl + 1)
  
  nD <- length(delta_vals); klF <- numeric(nD); klP <- numeric(nD)
  
  for (j in seq_along(delta_vals)) {
    delta   <- delta_vals[j]
    y_tilde <- max(0, y[t] + delta)
    a_att_j <- gamma * a_att_prev + y_tilde; b_att_j <- gamma * b_att_prev + 1
    r_att_j <- gamma * a_att_j;              p_att_j <- (gamma * b_att_j) / (gamma * b_att_j + 1)
    klF[j]  <- kl_gamma(a_att_j, b_att_j, a_cl, b_cl)
    klP[j]  <- kl_negbin(r_att_j, p_att_j, r_cl, p_cl, Kmax)
  }
  data.frame(delta = delta_vals, klF = klF, klP = klP,
             cost = cost_rate * delta_vals^2,
             utility = as.numeric(wF * klF + wP * klP - cost_rate * delta_vals^2))
}

cost_sens_grid <- exp(seq(log(0.001), log(50), length.out = 120))
sens_rows <- list()

for (cr in cost_sens_grid) {
  for (ar in archetypes) {
    df_tmp   <- one_step_wb_raw(t = t_eval, delta_vals = delta_vals, y = y,
                                a_clean_prev = a_prev, b_clean_prev = b_prev,
                                a_att_prev = a_prev, b_att_prev = b_prev,
                                gamma = gamma_nom, d_fixed = d_base,
                                wF = ar$wF, wP = ar$wP, cost_rate = cr, Kmax = Kmax_val)
    best_row <- df_tmp[which.max(df_tmp$utility), ]
    sens_rows[[length(sens_rows) + 1]] <- data.frame(
      cost_rate = cr, archetype = ar$label_short,
      delta_opt = best_row$delta, util_opt = best_row$utility,
      profitable = best_row$utility > 0, stringsAsFactors = FALSE)
  }
}
sens_cost_df <- bind_rows(sens_rows)
sens_cost_df$archetype <- factor(sens_cost_df$archetype, levels = c("SP", "FM"))

sp_sens <- sens_cost_df %>% filter(archetype == "SP") %>% arrange(cost_rate)
fm_sens <- sens_cost_df %>% filter(archetype == "FM") %>% arrange(cost_rate)

fm_profitable <- fm_sens %>% filter(profitable)
c_fm_thresh   <- if (nrow(fm_profitable) > 0) max(fm_profitable$cost_rate) else NA
sp_profitable <- sp_sens %>% filter(profitable)
c_sp_thresh   <- if (nrow(sp_profitable) > 0) max(sp_profitable$cost_rate) else NA

cat(sprintf("  c* (FM threshold) = %.4f\n", c_fm_thresh))
cat(sprintf("  c_SP threshold    = %.4f\n", c_sp_thresh))

cost_base <- if (!is.na(c_fm_thresh)) c_fm_thresh else 0.5
c_low  <- cost_base / 10
c_star <- cost_base
c_high <- if (!is.na(c_sp_thresh)) (cost_base + c_sp_thresh) / 2 else cost_base * 3

regime_df <- data.frame(
  xmin  = c(min(cost_sens_grid), c_fm_thresh, c_sp_thresh),
  xmax  = c(c_fm_thresh, c_sp_thresh, max(cost_sens_grid)),
  label = c("Both attack\n(unconstrained)", "SP only\n(profitable)", "No attack\n(deterred)"),
  fill  = c("#FDDBC7", "#D1E5F0", "#F0F0F0"))
regime_df <- regime_df[!is.na(regime_df$xmin) & !is.na(regime_df$xmax) &
                         regime_df$xmin < regime_df$xmax, ]

arch_colors <- setNames(c(archetypes$SP$color, archetypes$FM$color), c("SP", "FM"))

p_s1_0 <- ggplot(sens_cost_df, aes(x = cost_rate, y = delta_opt,
                                   color = archetype, linetype = profitable)) +
  {if (nrow(regime_df) > 0)
    geom_rect(data = regime_df,
              aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = label),
              inherit.aes = FALSE, alpha = 0.25)} +
  scale_fill_manual(values = setNames(regime_df$fill, regime_df$label), name = "Regime") +
  geom_line(linewidth = 1.2) + geom_point(size = 1.5) +
  geom_vline(xintercept = cost_base, linetype = "dotted", color = "grey20", linewidth = 1.1) +
  annotate("text", x = cost_base * 1.15, y = max(delta_vals) * 0.88,
           label = paste0("c* = ", round(cost_base, 3), "\n(FM threshold)"),
           size = 3.2, hjust = 0, color = "grey20") +
  geom_vline(xintercept = c_low,  linetype = "dashed", color = "#984EA3", linewidth = 0.8) +
  geom_vline(xintercept = c_high, linetype = "dashed", color = "#FF7F00", linewidth = 0.8) +
  annotate("text", x = c_low,  y = max(delta_vals) * 0.55, label = "c_low\n",
           size = 2.8, color = "#984EA3", hjust = -0.1) +
  annotate("text", x = c_high, y = max(delta_vals) * 0.55, label = "c_high\n",
           size = 2.8, color = "#FF7F00", hjust = -0.1) +
  scale_x_log10(breaks = c(0.001, 0.01, 0.1, 1, 10, 50),
                labels = c("0.001","0.01","0.1","1","10","50")) +
  scale_color_manual(values = arch_colors, name = NULL) +
  scale_linetype_manual(values = c("TRUE" = "solid", "FALSE" = "dashed"),
                        labels = c("TRUE" = "Profitable (U*>0)", "FALSE" = "Unprofitable (U*<=0)"),
                        name = NULL) +
  labs(title = "Cost Sensitivity: Optimal Attack Size vs Cost Coefficient",
       x = "Cost coefficient c (log scale)", y = expression(delta*"*")) +
  theme_ara

print(p_s1_0)
ggsave("S1_0_cost_sensitivity.pdf", p_s1_0, width = 11, height = 5)

# =============================================================================
# S1.1  UTILITY PROFILE SENSITIVITY ACROSS THREE COST REGIMES (Figure 2)
#
# Panels A, B, C correspond to c_low, c*, and c_high respectively.
# Each panel shows the utility decomposition for SP (left) and FM (right).
# =============================================================================
cat("Running S1.1 ...\n")

make_s11_df <- function(cost_val, regime_label) {
  bind_rows(lapply(archetypes, function(ar) {
    df <- one_step_wb_raw(t = t_eval, delta_vals = delta_vals, y = y,
                          a_clean_prev = a_prev, b_clean_prev = b_prev,
                          a_att_prev = a_prev, b_att_prev = b_prev,
                          gamma = gamma_nom, d_fixed = d_base,
                          wF = ar$wF, wP = ar$wP, cost_rate = cost_val, Kmax = Kmax_val)
    df$archetype    <- ar$label_short
    df$arch_label   <- ar$label_short
    df$cost_val     <- cost_val
    df$regime_label <- regime_label
    df$arch_color   <- ar$color
    df
  }))
}

cost_regimes <- list(
  list(c = c_low,  label = paste0("A: c_low = ", round(c_low, 4),
                                  "\n(Below FM threshold -- unconstrained)")),
  list(c = c_star, label = paste0("B: c* = ", sprintf("%.3f", round(c_star, 3)),
                                  "\n(FM threshold -- target regime)")),
  list(c = c_high, label = paste0("C: c_high = ", round(c_high, 4),
                                  "\n(Above FM threshold -- SP only)")))

s1_1_all <- bind_rows(lapply(cost_regimes, function(cr) make_s11_df(cr$c, cr$label)))
s1_1_all$regime_label <- factor(s1_1_all$regime_label, levels = sapply(cost_regimes, `[[`, "label"))
s1_1_all$arch_label   <- factor(s1_1_all$arch_label,
                                levels = c(archetypes$SP$label_short, archetypes$FM$label_short))

s1_1_opt <- s1_1_all %>%
  group_by(regime_label, arch_label, archetype, arch_color, cost_val) %>%
  slice_max(utility, n = 1, with_ties = FALSE) %>% ungroup()

s1_1_long <- s1_1_all %>%
  pivot_longer(cols = c(klF, klP, cost, utility), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric, klF = "Filtering KL", klP = "Predictive KL",
                         cost = "Cost", utility = "Net Utility"))

p_s1_1 <- ggplot(s1_1_long, aes(x = delta, y = value, color = metric, linetype = metric)) +
  geom_line(linewidth = 0.85) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  geom_vline(data = s1_1_opt, aes(xintercept = delta, color = NULL),
             color = s1_1_opt$arch_color, linetype = "dotted", linewidth = 1.0, show.legend = FALSE) +
  geom_text(data = s1_1_opt,
            aes(x = delta, y = Inf, label = paste0("delta*=", delta),
                group = interaction(regime_label, arch_label)),
            color = s1_1_opt$arch_color, vjust = 1.5, hjust = -0.1, size = 3.0, inherit.aes = FALSE) +
  facet_grid(regime_label ~ arch_label, scales = "free_y") +
  scale_color_manual(values = c("Filtering KL" = "#E41A1C", "Predictive KL" = "#377EB8",
                                "Cost" = "#4DAF4A", "Net Utility" = "#FF7F00")) +
  scale_linetype_manual(values = c("Filtering KL" = "solid", "Predictive KL" = "solid",
                                   "Cost" = "dashed", "Net Utility" = "solid")) +
  labs(title = "Attacker Utility Profile Sensitivity: Three Cost Regimes x Two Archetypes",
       x = expression(delta), y = "Value", color = NULL, linetype = NULL) +
  theme_ara + theme(strip.text = element_text(size = 9))

print(p_s1_1)
ggsave("S1_1_utility_profile.pdf", p_s1_1, width = 13, height = 11)

s1_1_opt_cstar <- s1_1_opt %>% filter(cost_val == c_star)

# =============================================================================
# S1.2  POSTERIOR DISTRIBUTION SHIFTS (Figure 3)
#
# 2x2 layout: rows = archetype (SP, FM); columns = NegBin predictive PMF
# and Gamma filtering density. Uses delta* from the calibrated cost c*.
# =============================================================================
cat("Running S1.2 ...\n")

k_grid   <- 0:50
lam_max  <- (a_cl_base / b_cl_base) * 5
lam_grid <- seq(0, lam_max, length.out = 600)

make_s12_rows <- function(ar) {
  ds <- s1_1_opt_cstar$delta[s1_1_opt_cstar$archetype == ar$label_short]
  if (length(ds) == 0) {
    res <- one_step_wb_raw(t = t_eval, delta_vals = delta_vals, y = y,
                           a_clean_prev = a_prev, b_clean_prev = b_prev,
                           a_att_prev = a_prev, b_att_prev = b_prev,
                           gamma = gamma_nom, d_fixed = d_base,
                           wF = ar$wF, wP = ar$wP, cost_rate = cost_base, Kmax = Kmax_val)
    ds <- res$delta[which.max(res$utility)]
  }
  ds <- as.integer(ds[1])
  y_til  <- max(0, y[t_eval] + ds)
  a_att  <- gamma_nom * a_prev + y_til; b_att <- gamma_nom * b_prev + 1
  r_att  <- gamma_nom * a_att;          p_att <- (gamma_nom * b_att) / (gamma_nom * b_att + 1)
  att_lbl <- paste0("Attacked (delta* = ", ds, ")")
  
  pred_rows <- data.frame(
    k = rep(k_grid, 2),
    prob = c(dnbinom(k_grid, size = r_cl_base, prob = 1 - p_cl_base),
             dnbinom(k_grid, size = r_att, prob = 1 - p_att)),
    scenario = rep(c("Clean", att_lbl), each = length(k_grid)),
    archetype = ar$label_short, dist_type = "Predictive PMF",
    clr_clean = "#888888", clr_att = ar$color, stringsAsFactors = FALSE)
  
  filt_rows <- data.frame(
    lam = rep(lam_grid, 2),
    dens = c(dgamma(lam_grid, shape = a_cl_base, rate = b_cl_base),
             dgamma(lam_grid, shape = a_att, rate = b_att)),
    scenario = rep(c("Clean", att_lbl), each = length(lam_grid)),
    archetype = ar$label_short, dist_type = "Filtering Density",
    clr_clean = "#888888", clr_att = ar$color, stringsAsFactors = FALSE)
  
  list(pred = pred_rows, filt = filt_rows, delta_star = ds, att_lbl = att_lbl,
       a_att = a_att, b_att = b_att, r_att = r_att, p_att = p_att,
       color = ar$color, archetype = ar$label_short)
}

s12_sp <- make_s12_rows(archetypes$SP)
s12_fm <- make_s12_rows(archetypes$FM)

make_pred_panel <- function(s12, title_str) {
  att_lbl  <- s12$att_lbl
  clr_vals <- setNames(c("#888888", s12$color), c("Clean", att_lbl))
  ggplot(s12$pred, aes(x = k, y = prob, color = scenario, group = scenario)) +
    geom_line(linewidth = 1.0) + geom_point(size = 1.2) +
    scale_color_manual(values = clr_vals) +
    labs(title = title_str, x = expression(Y[t+1]), y = "Probability", color = NULL) +
    theme_ara + theme(legend.position = "right")
}

make_filt_panel <- function(s12, title_str) {
  att_lbl  <- s12$att_lbl
  clr_vals <- setNames(c("#888888", s12$color), c("Clean", att_lbl))
  ggplot(s12$filt, aes(x = lam, y = dens, color = scenario, group = scenario)) +
    geom_line(linewidth = 1.0) + scale_color_manual(values = clr_vals) +
    labs(title = title_str, x = expression(lambda[t]), y = "Density", color = NULL) +
    theme_ara + theme(legend.position = "right")
}

p_sp_pred <- make_pred_panel(s12_sp, paste0("SP: Predictive Shift  (delta* = ", s12_sp$delta_star, ")"))
p_sp_filt <- make_filt_panel(s12_sp, paste0("SP: Filtering Shift  (delta* = ", s12_sp$delta_star, ")"))
p_fm_pred <- make_pred_panel(s12_fm, paste0("FM: Predictive Shift  (delta* = ", s12_fm$delta_star, ")"))
p_fm_filt <- make_filt_panel(s12_fm, paste0("FM: Filtering Shift  (delta* = ", s12_fm$delta_star, ")"))

p_s1_2 <- (p_sp_pred | p_sp_filt) / (p_fm_pred | p_fm_filt) +
  plot_annotation(title = "Posterior Distribution Shifts by Attacker Archetype")

print(p_s1_2)
ggsave("S1_2_distribution_shifts.pdf", p_s1_2, width = 13, height = 9)

# =============================================================================
# S1.3  SENSITIVITY HEATMAP AND THRESHOLD PLOT (Supplemental Figure 1)
#
# Sweeps wF in [0,1] and cost rate c jointly to show profitable attack regions
# and optimal delta* across the full (wF, c) parameter space.
# =============================================================================
cat("Running S1.3 ...\n")

wf_grid   <- seq(0, 1, by = 0.025)
cost_grid <- exp(seq(log(0.01), log(1.5), length.out = 40))

heatmap_rows <- list()
for (wf_val in wf_grid) {
  for (cr_val in cost_grid) {
    df_tmp <- one_step_wb_raw(t = t_eval, delta_vals = delta_vals, y = y,
                              a_clean_prev = a_prev, b_clean_prev = b_prev,
                              a_att_prev = a_prev, b_att_prev = b_prev,
                              gamma = gamma_nom, d_fixed = d_base,
                              wF = wf_val, wP = 1 - wf_val, cost_rate = cr_val, Kmax = Kmax_val)
    best_row <- df_tmp[which.max(df_tmp$utility), ]
    heatmap_rows[[length(heatmap_rows) + 1]] <- data.frame(
      wF = wf_val, cost_rate = cr_val, delta_opt = best_row$delta, util_opt = best_row$utility)
  }
}
heat_df <- bind_rows(heatmap_rows)

p_heat <- ggplot(heat_df, aes(x = wF, y = cost_rate, fill = delta_opt)) +
  geom_tile() + scale_fill_viridis_c(name = expression(delta*"*"), option = "C") +
  scale_y_log10(breaks = c(0.01, 0.05, 0.1, 0.5, 1.0),
                labels = c("0.01","0.05","0.1","0.5","1.0")) +
  labs(title = "Heatmap of Optimal Attack Size",
       x = expression(w[F]), y = "Cost rate (log scale)") + theme_ara

p_thresh <- ggplot(heat_df, aes(x = wF, y = cost_rate, fill = util_opt <= 0)) +
  geom_tile() +
  scale_fill_manual(values = c("TRUE" = "#2166AC", "FALSE" = "#F4A582"),
                    labels = c("TRUE" = "No attack profitable (U<=0)", "FALSE" = "Attack profitable (U>0)"),
                    name = NULL) +
  scale_y_log10(breaks = c(0.01, 0.05, 0.1, 0.5, 1.0),
                labels = c("0.01","0.05","0.1","0.5","1.0")) +
  labs(title = "Threshold: Profitable Attack Region",
       x = expression(w[F]), y = "Cost rate (log scale)") + theme_ara

p_s1_3 <- p_heat + p_thresh
print(p_s1_3)
ggsave("S1_3_heatmap_threshold.pdf", p_s1_3, width = 12, height = 5)

# =============================================================================
# S1.4  GRAY-BOX VS. WHITE-BOX UTILITY CURVES (Figure 4)
#
# Gray-box: Monte Carlo over gamma ~ Uniform{0.7, 0.8, 0.9} and theta ~ Beta,
# using raw KL utility. White-box: deterministic at gamma = 0.8.
# =============================================================================
cat("Running S1.4 ...\n")

one_step_gb <- function(t, delta_vals, y,
                        a_clean_prev, b_clean_prev,
                        a_att_prev, b_att_prev,
                        gamma_vec = c(0.7, 0.8, 0.9),
                        gamma_probs = rep(1/3, 3),
                        d_fixed = 0, wF = 0.5, wP = 0.5,
                        cost_rate = 0.1, Kmax = 500, n_samples = 1000) {
  cost_rate   <- as.numeric(cost_rate)[1]
  wF          <- as.numeric(wF)[1]; wP <- as.numeric(wP)[1]
  nD          <- length(delta_vals)
  gamma_probs <- gamma_probs / sum(gamma_probs)
  util_mat    <- matrix(NA, nD, n_samples)
  klF_mat     <- matrix(NA, nD, n_samples)
  klP_mat     <- matrix(NA, nD, n_samples)
  
  for (s in 1:n_samples) {
    gam_s  <- sample(gamma_vec, 1, prob = gamma_probs)
    a_cl_s <- gam_s * a_clean_prev + y[t]; b_cl_s <- gam_s * b_clean_prev + 1
    r_cl_s <- gam_s * a_cl_s;              p_cl_s <- (gam_s * b_cl_s) / (gam_s * b_cl_s + 1)
    klF_raw <- numeric(nD); klP_raw <- numeric(nD)
    
    for (j in seq_along(delta_vals)) {
      delta   <- delta_vals[j]
      alpha_s <- alpha_fun(delta, d_fixed); beta_s <- beta_fun(delta, d_fixed)
      theta_s <- rbeta(1, alpha_s, beta_s); success <- rbinom(1, 1, theta_s)
      y_tilde <- if (success == 1) max(0, y[t] + delta) else y[t]
      a_att_s <- gam_s * a_att_prev + y_tilde; b_att_s <- gam_s * b_att_prev + 1
      r_att_s <- gam_s * a_att_s;              p_att_s <- (gam_s * b_att_s) / (gam_s * b_att_s + 1)
      klF_raw[j] <- kl_gamma(a_att_s, b_att_s, a_cl_s, b_cl_s)
      klP_raw[j] <- kl_negbin(r_att_s, p_att_s, r_cl_s, p_cl_s, Kmax)
    }
    klF_mat[, s]  <- klF_raw; klP_mat[, s] <- klP_raw
    util_mat[, s] <- as.numeric(wF * klF_raw + wP * klP_raw - cost_rate * delta_vals^2)
  }
  
  data.frame(delta = delta_vals,
             util_mean = rowMeans(util_mat),
             util_lo   = apply(util_mat, 1, quantile, 0.025),
             util_hi   = apply(util_mat, 1, quantile, 0.975),
             klF_mean  = rowMeans(klF_mat),
             klP_mean  = rowMeans(klP_mat))
}

make_s14_panel <- function(ar) {
  gb <- one_step_gb(t = t_eval, delta_vals = delta_vals, y = y,
                    a_clean_prev = a_prev, b_clean_prev = b_prev,
                    a_att_prev = a_prev, b_att_prev = b_prev,
                    gamma_vec = gamma_vec, gamma_probs = gamma_probs,
                    d_fixed = d_base, wF = ar$wF, wP = ar$wP,
                    cost_rate = cost_base, Kmax = Kmax_val, n_samples = n_samp)
  wb <- one_step_wb_raw(t = t_eval, delta_vals = delta_vals, y = y,
                        a_clean_prev = a_prev, b_clean_prev = b_prev,
                        a_att_prev = a_prev, b_att_prev = b_prev,
                        gamma = gamma_nom, d_fixed = d_base,
                        wF = ar$wF, wP = ar$wP, cost_rate = cost_base, Kmax = Kmax_val)
  ds_gb <- gb$delta[which.max(gb$util_mean)]
  ds_wb <- wb$delta[which.max(wb$utility)]
  y_top <- max(gb$util_hi, wb$utility, na.rm = TRUE)
  y_range <- y_top - min(gb$util_lo, wb$utility, na.rm = TRUE)
  
  ggplot(gb, aes(x = delta)) +
    geom_ribbon(aes(ymin = util_lo, ymax = util_hi), fill = ar$color, alpha = 0.20) +
    geom_line(aes(y = util_mean, color = "Gray-box (mean)"), linewidth = 1.1) +
    geom_line(data = wb, aes(y = utility, color = "White-box"), linewidth = 1.1, linetype = "dashed") +
    geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
    geom_vline(xintercept = ds_gb, color = ar$color, linetype = "solid", linewidth = 0.9) +
    {if (ds_wb != ds_gb)
      geom_vline(xintercept = ds_wb, color = "black", linetype = "dashed", linewidth = 0.7)} +
    annotate("text", x = ds_gb + 0.5, y = y_top - 0.05 * y_range,
             label = paste0("delta*(GB)=", ds_gb), size = 3.3, color = ar$color, hjust = 0) +
    annotate("text", x = ds_wb + 0.5, y = y_top - 0.18 * y_range,
             label = paste0("delta*(WB)=", ds_wb), size = 3.3, color = "black", hjust = 0) +
    scale_color_manual(values = c("Gray-box (mean)" = ar$color, "White-box" = "black"), name = NULL) +
    labs(title = paste0("Gray-Box vs White-Box: ", ar$label_short),
         x = expression(delta), y = "Utility", color = NULL) + theme_ara
}

p_s1_4_sp <- make_s14_panel(archetypes$SP)
p_s1_4_fm <- make_s14_panel(archetypes$FM)
p_s1_4 <- p_s1_4_sp | p_s1_4_fm
print(p_s1_4)
ggsave("S1_4_graybox_utility.pdf", p_s1_4, width = 14, height = 5)

# =============================================================================
# S1.5  ATTACK SUCCESS PROBABILITY BY DEFENSE LEVEL (Figure 5)
#
# Plots E[theta | delta, d] = mu(delta, d) for d in {0, 2, 5, 10} with 95%
# Beta credible bands, annotating SP and FM archetype-optimal deltas.
# =============================================================================
cat("Running S1.5 ...\n")

d_levels   <- c(0, 2, 5, 10)
delta_fine <- seq(min(delta_vals), max(delta_vals), by = 0.2)

succ_list <- lapply(d_levels, function(dval) {
  mu_vals  <- sapply(delta_fine, mu_fun, d = dval)
  al_vals  <- sapply(delta_fine, alpha_fun, d = dval)
  be_vals  <- sapply(delta_fine, beta_fun, d = dval)
  var_vals <- (al_vals * be_vals) / ((al_vals + be_vals)^2 * (al_vals + be_vals + 1))
  sd_vals  <- sqrt(var_vals)
  data.frame(delta = delta_fine, mu = mu_vals, sd = sd_vals,
             lo = pmax(0, mu_vals - 1.96 * sd_vals),
             hi = pmin(1, mu_vals + 1.96 * sd_vals), d = dval)
})
succ_df <- bind_rows(succ_list)
succ_df$d_label <- factor(paste0("d = ", succ_df$d), levels = paste0("d = ", d_levels))

ds_sp_s15 <- s1_1_opt_cstar$delta[s1_1_opt_cstar$archetype == "SP"]
ds_fm_s15 <- s1_1_opt_cstar$delta[s1_1_opt_cstar$archetype == "FM"]

p_s1_5 <- ggplot(succ_df, aes(x = delta, y = mu, color = d_label, fill = d_label)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = ds_sp_s15, color = archetypes$SP$color, linetype = "dotted", linewidth = 1.1) +
  geom_vline(xintercept = ds_fm_s15, color = archetypes$FM$color, linetype = "dotted", linewidth = 1.1) +
  annotate("text", x = ds_sp_s15, y = 0.90,
           label = paste0("SP\nd*=", ds_sp_s15),
           color = archetypes$SP$color, size = 3.2, hjust = -0.1, lineheight = 0.85) +
  annotate("text", x = ds_fm_s15, y = 0.78,
           label = paste0("FM\nd*=", ds_fm_s15),
           color = archetypes$FM$color, size = 3.2, hjust = -0.1, lineheight = 0.85) +
  scale_color_viridis_d(option = "D", end = 0.85) +
  scale_fill_viridis_d(option  = "D", end = 0.85) +
  labs(title = "Attack Success Probability vs delta, by Defense Level",
       x = expression(delta), y = expression(E[theta]), color = NULL, fill = NULL) +
  theme_ara

print(p_s1_5)
ggsave("S1_5_attack_success.pdf", p_s1_5, width = 10, height = 5)

# =============================================================================
# S1.6  SENSITIVITY OF ATTACK SUCCESS TO LOGISTIC PARAMETERS (Supplemental Figure 2)
#
# Three panels: varying s1 (sensitivity to delta), s2 (deterrence), and
# c_theta (Beta concentration) at d = 2.
# =============================================================================
cat("Running S1.6 ...\n")

s1_vals <- c(0.1, 0.5, 1.0)
s2_vals <- c(-0.4, -0.2, 0.0)
c1_vals <- c(1, 10, 100)

df_s1 <- bind_rows(lapply(s1_vals, function(s1) {
  data.frame(delta = delta_vals,
             mu = sapply(delta_vals, function(del) 1/(1+exp(-(a011 + s1*del + a211*2)))),
             param_val = as.character(s1), param = "s[1] (sensitivity)")
}))

df_s2 <- bind_rows(lapply(s2_vals, function(s2) {
  data.frame(delta = delta_vals,
             mu = sapply(delta_vals, function(del) 1/(1+exp(-(a011 + a111*del + s2*2)))),
             param_val = as.character(s2), param = "s[2] (deterrence)")
}))

df_c1 <- bind_rows(lapply(c1_vals, function(cc) {
  mu_v <- sapply(delta_vals, function(del) 1/(1+exp(-(a011 + a111*del + a211*2))))
  al_v <- cc * mu_v; be_v <- cc * (1 - mu_v)
  sd_v <- sqrt((al_v * be_v) / ((al_v + be_v)^2 * (al_v + be_v + 1)))
  data.frame(delta = delta_vals, mu = mu_v, sd = sd_v,
             param_val = as.character(cc), param = "c[theta] (concentration)")
}))

p_s1 <- ggplot(df_s1, aes(x = delta, y = mu, color = param_val)) +
  geom_line(linewidth = 1.1) + scale_color_brewer(palette = "Set1", name = expression(s[1])) +
  labs(title = expression("Vary "*s[1]*" (sensitivity to "*delta*")"),
       x = expression(delta), y = expression(E[theta])) + ylim(0, 1) + theme_ara

p_s2 <- ggplot(df_s2, aes(x = delta, y = mu, color = param_val)) +
  geom_line(linewidth = 1.1) + scale_color_brewer(palette = "Set2", name = expression(s[2])) +
  labs(title = expression("Vary "*s[2]*" (deterrence)"),
       x = expression(delta), y = expression(E[theta])) + ylim(0, 1) + theme_ara

p_c1 <- ggplot(df_c1, aes(x = delta, y = mu, color = param_val,
                          ymin = pmax(0, mu - sd), ymax = pmin(1, mu + sd), fill = param_val)) +
  geom_ribbon(alpha = 0.2, color = NA) + geom_line(linewidth = 1.1) +
  scale_color_brewer(palette = "Set3", name = expression(c[theta])) +
  scale_fill_brewer(palette  = "Set3", name = expression(c[theta])) +
  labs(title = expression("Vary "*c[theta]*" (concentration)"),
       x = expression(delta), y = expression(E[theta])) + ylim(0, 1) + theme_ara

p_s1_6 <- p_s1 + p_s2 + p_c1 +
  plot_annotation(title = "Sensitivity of Attack Success Probability to Parameters")

print(p_s1_6)
ggsave("S1_6_success_sensitivity.pdf", p_s1_6, width = 14, height = 5)

# =============================================================================
# SEQUENTIAL ATTACK ALGORITHM (Algorithm 2, Appendix A)
#
# Myopic gray-box formulation: at each period tau, the attacker solves a
# one-step expected-utility problem using the currently attacked filter state.
# Expected utility uses mu_fun (logistic mean) rather than a single theta draw
# to avoid high-variance optimization under strong defense.
# =============================================================================

run_sequential_attack <- function(
    y, t_start,
    a_clean_all, b_clean_all,
    H = 10, d_seq = rep(0, H),
    gamma_vec = c(0.7, 0.8, 0.9),
    gamma_probs = rep(1/3, 3),
    delta_vals = -5:20,
    wF = 0.5, wP = 0.5,
    cost_rate = 0.1, Kmax = 500,
    n_samples = 500) {
  
  gamma_probs <- gamma_probs / sum(gamma_probs)
  delta_opt_mat <- matrix(NA, n_samples, H)
  klF_mat       <- matrix(NA, n_samples, H)
  klP_mat       <- matrix(NA, n_samples, H)
  util_mat      <- matrix(NA, n_samples, H)
  a_att_mat     <- matrix(NA, n_samples, H)
  b_att_mat     <- matrix(NA, n_samples, H)
  a_cl_mat      <- matrix(NA, n_samples, H)
  
  for (s in 1:n_samples) {
    gam_s      <- sample(gamma_vec, 1, prob = gamma_probs)
    a_att_prev <- a_clean_all[t_start - 1]; b_att_prev <- b_clean_all[t_start - 1]
    a_cl_prev  <- a_clean_all[t_start - 1]; b_cl_prev  <- b_clean_all[t_start - 1]
    
    for (k in 1:H) {
      tau <- t_start + k - 1
      if (tau > length(y)) break
      d_k    <- d_seq[k]
      a_cl_k <- gam_s * a_cl_prev + y[tau]; b_cl_k <- gam_s * b_cl_prev + 1
      r_cl_k <- gam_s * a_cl_k;             p_cl_k <- (gam_s * b_cl_k) / (gam_s * b_cl_k + 1)
      a_cl_mat[s, k] <- a_cl_k
      
      best_util <- -Inf; best_delta <- delta_vals[1]; best_klF <- 0; best_klP <- 0
      
      for (del in delta_vals) {
        mu_d    <- mu_fun(del, d_k)
        y_til_s <- max(0, y[tau] + del)
        a_att_s <- gam_s * a_att_prev + y_til_s; b_att_s <- gam_s * b_att_prev + 1
        r_att_s <- gam_s * a_att_s;              p_att_s <- (gam_s * b_att_s) / (gam_s * b_att_s + 1)
        a_att_f <- gam_s * a_att_prev + y[tau];  b_att_f <- gam_s * b_att_prev + 1
        r_att_f <- gam_s * a_att_f;              p_att_f <- (gam_s * b_att_f) / (gam_s * b_att_f + 1)
        
        klF_s <- kl_gamma(a_att_s, b_att_s, a_cl_k, b_cl_k)
        klP_s <- kl_negbin(r_att_s, p_att_s, r_cl_k, p_cl_k, Kmax)
        klF_f <- kl_gamma(a_att_f, b_att_f, a_cl_k, b_cl_k)
        klP_f <- kl_negbin(r_att_f, p_att_f, r_cl_k, p_cl_k, Kmax)
        
        klF_k  <- mu_d * klF_s + (1 - mu_d) * klF_f
        klP_k  <- mu_d * klP_s + (1 - mu_d) * klP_f
        util_k <- as.numeric(wF * klF_k + wP * klP_k - cost_rate * del^2)
        
        if (util_k > best_util) {
          best_util <- util_k; best_delta <- del; best_klF <- klF_k; best_klP <- klP_k
        }
      }
      
      mu_best    <- mu_fun(best_delta, d_k)
      theta_best <- rbeta(1, alpha_fun(best_delta, d_k), beta_fun(best_delta, d_k))
      succ_best  <- rbinom(1, 1, theta_best)
      y_til_best <- if (succ_best == 1) max(0, y[tau] + best_delta) else y[tau]
      a_att_best <- gam_s * a_att_prev + y_til_best; b_att_best <- gam_s * b_att_prev + 1
      
      delta_opt_mat[s, k] <- best_delta; klF_mat[s, k] <- best_klF
      klP_mat[s, k]       <- best_klP;   util_mat[s, k] <- best_util
      a_att_mat[s, k]     <- a_att_best; b_att_mat[s, k] <- b_att_best
      
      a_att_prev <- a_att_best; b_att_prev <- b_att_best
      a_cl_prev  <- a_cl_k;    b_cl_prev  <- b_cl_k
    }
  }
  
  list(delta_opt = delta_opt_mat, klF = klF_mat, klP = klP_mat,
       util = util_mat, a_att = a_att_mat, b_att = b_att_mat, a_cl = a_cl_mat)
}

# =============================================================================
# S2.1  CLEAN VS. ATTACKED PGSS FILTER TRAJECTORY (Figure 6)
#
# Two rows (SP, FM) at d = 0 and d = 10. SP cascade compounds more than FM,
# consistent with the SP objective of maximizing filtering KL.
# =============================================================================
cat("Running S2.1 ...\n")

stopifnot(
  "a211 is stale -- restart R and re-run from the top" = isTRUE(all.equal(a211, -1.0)),
  "cost_base not set -- S1.0 must run first" = !is.na(cost_base) && cost_base > 0)

t_start_s2 <- n_train - 19
H_s2       <- 20
n_samp_s2  <- 300

run_s21 <- function(ar, d_val) {
  dvs <- if (ar$label_short == "FM") delta_vals_nonneg else delta_vals
  run_sequential_attack(y = y, t_start = t_start_s2,
                        a_clean_all = a_clean, b_clean_all = b_clean,
                        H = H_s2, d_seq = rep(d_val, H_s2),
                        gamma_vec = gamma_vec, gamma_probs = gamma_probs,
                        delta_vals = dvs, wF = ar$wF, wP = ar$wP,
                        cost_rate = cost_base, Kmax = Kmax_val, n_samples = n_samp_s2)
}

set.seed(42); res_sp_d0  <- run_s21(archetypes$SP, 0)
set.seed(42); res_sp_d10 <- run_s21(archetypes$SP, 10)
set.seed(42); res_fm_d0  <- run_s21(archetypes$FM, 0)
set.seed(42); res_fm_d10 <- run_s21(archetypes$FM, 10)

periods         <- t_start_s2:(t_start_s2 + H_s2 - 1)
clean_mean_true <- a_clean[periods] / b_clean[periods]

make_traj_df <- function(res_d0, res_d10, archetype_label) {
  rate_fn     <- function(res) res$a_att / res$b_att
  att_d0_mat  <- rate_fn(res_d0)
  att_d10_mat <- rate_fn(res_d10)
  data.frame(
    period    = rep(periods, 3),
    mean_rate = c(clean_mean_true, colMeans(att_d0_mat), colMeans(att_d10_mat)),
    lo        = c(rep(NA, H_s2),
                  apply(att_d0_mat,  2, quantile, 0.025),
                  apply(att_d10_mat, 2, quantile, 0.025)),
    hi        = c(rep(NA, H_s2),
                  apply(att_d0_mat,  2, quantile, 0.975),
                  apply(att_d10_mat, 2, quantile, 0.975)),
    scenario  = rep(c("Clean (no attack)", "Attacked (d=0)", "Attacked (d=10)"), each = H_s2),
    archetype = archetype_label)
}

traj_sp <- make_traj_df(res_sp_d0, res_sp_d10, archetypes$SP$label_short)
traj_fm <- make_traj_df(res_fm_d0, res_fm_d10, archetypes$FM$label_short)
traj_df <- bind_rows(traj_sp, traj_fm)
traj_df$archetype <- factor(traj_df$archetype,
                            levels = c(archetypes$SP$label_short, archetypes$FM$label_short))

obs_df       <- data.frame(period = periods, y_obs = y[periods])
traj_ribbon  <- traj_df[traj_df$scenario %in% c("Attacked (d=0)", "Attacked (d=10)") &
                          !is.na(traj_df$lo), ]

scenario_colors <- c("Clean (no attack)" = "grey40", "Attacked (d=0)" = "#E41A1C",
                     "Attacked (d=10)"   = "#FF7F00")

p_s2_1 <- ggplot() +
  geom_ribbon(data = traj_ribbon,
              aes(x = period, ymin = lo, ymax = hi, fill = scenario,
                  group = interaction(scenario, archetype)),
              alpha = 0.28, color = NA) +
  geom_line(data = traj_df,
            aes(x = period, y = mean_rate, color = scenario,
                group = interaction(scenario, archetype)), linewidth = 1.2) +
  geom_point(data = obs_df, aes(x = period, y = y_obs),
             shape = 16, size = 1.3, color = "grey30", inherit.aes = FALSE) +
  facet_wrap(~archetype, ncol = 1,
             labeller = labeller(archetype = c("SP" = "SP  (wF=1, wP=0)", "FM" = "FM  (wF=0, wP=1)"))) +
  scale_color_manual(values = scenario_colors, name = NULL) +
  scale_fill_manual(values  = scenario_colors, guide = "none") +
  labs(title = "Clean vs Attacked PGSS Filter Trajectory by Archetype",
       x = "Time period t", y = expression(E[lambda[t]])) + theme_ara

print(p_s2_1)
ggsave("S2_1_filter_trajectory.pdf", p_s2_1, width = 10, height = 10)

# =============================================================================
# S2.2  CUMULATIVE KL DIVERGENCE OVER HORIZON (Figure 7)
#
# Both archetypes, horizons H in {5, 10, 20}. Rows: Filtering KL (top) and
# Predictive KL (bottom). FM: predictive KL dominates; SP: filtering KL dominates.
# =============================================================================
cat("Running S2.2 ...\n")

H_vals_s22   <- c(5, 10, 20)
arc_list_s22 <- list(list(ar = archetypes$FM, dvals = delta_vals_nonneg),
                     list(ar = archetypes$SP, dvals = delta_vals))
cum_kl_list  <- list()

for (arc_spec in arc_list_s22) {
  ar <- arc_spec$ar; dvs <- arc_spec$dvals
  for (H_val in H_vals_s22) {
    res_tmp <- run_sequential_attack(y = y, t_start = t_start_s2,
                                     a_clean_all = a_clean, b_clean_all = b_clean,
                                     H = H_val, d_seq = rep(0, H_val),
                                     gamma_vec = gamma_vec, gamma_probs = gamma_probs,
                                     delta_vals = dvs, wF = ar$wF, wP = ar$wP,
                                     cost_rate = cost_base, Kmax = Kmax_val, n_samples = n_samp_s2)
    cum_klF <- t(apply(res_tmp$klF, 1, cumsum))
    cum_klP <- t(apply(res_tmp$klP, 1, cumsum))
    cum_kl_list[[length(cum_kl_list) + 1]] <- data.frame(
      k           = rep(1:H_val, 2),
      cum_kl_mean = c(colMeans(cum_klF), colMeans(cum_klP)),
      cum_kl_lo   = c(apply(cum_klF, 2, quantile, 0.025), apply(cum_klP, 2, quantile, 0.025)),
      cum_kl_hi   = c(apply(cum_klF, 2, quantile, 0.975), apply(cum_klP, 2, quantile, 0.975)),
      type        = rep(c("Filtering KL", "Predictive KL"), each = H_val),
      H = H_val, archetype = ar$label_short)
  }
}
cum_kl_df <- bind_rows(cum_kl_list)
cum_kl_df$H_label   <- factor(paste0("H = ", cum_kl_df$H), levels = c("H = 5","H = 10","H = 20"))
cum_kl_df$archetype <- factor(cum_kl_df$archetype, levels = c("FM", "SP"))

p_s2_2 <- ggplot(cum_kl_df, aes(x = k, y = cum_kl_mean, color = H_label, fill = H_label)) +
  geom_ribbon(aes(ymin = cum_kl_lo, ymax = cum_kl_hi), alpha = 0.25, color = NA) +
  geom_line(linewidth = 1.1) +
  facet_grid(type ~ archetype, scales = "free_y",
             labeller = labeller(archetype = c("FM" = "FM  (wF=0, wP=1)", "SP" = "SP  (wF=1, wP=0)"),
                                 type = label_value)) +
  scale_color_viridis_d(option = "C", end = 0.85) +
  scale_fill_viridis_d(option  = "C", end = 0.85) +
  labs(title = "Cumulative KL Divergence over Attack Horizon by Archetype",
       x = "Periods elapsed (k)", y = "Cumulative KL", color = NULL, fill = NULL) + theme_ara

print(p_s2_2)
ggsave("S2_2_cumulative_KL.pdf", p_s2_2, width = 12, height = 7)

# =============================================================================
# S2.4  CUMULATIVE ATTACKER UTILITY VS. HORIZON H (Figure 8)
#
# FM archetype with non-negative perturbations, defense levels d in {0,2,5,10}.
# Annotates the diminishing-returns threshold H_dr where marginal gain falls
# below 10% of the initial slope.
# =============================================================================
cat("Running S2.4 ...\n")

stopifnot("a211 stale in S2.4" = isTRUE(all.equal(a211, -1.0)))

set.seed(42)
H_grid_s24 <- c(3, 5, 10, 15, 20)
d_grid_s24 <- c(0, 2, 5, 10)
util_H_list <- list()

for (H_val in H_grid_s24) {
  for (d_val in d_grid_s24) {
    res_tmp <- run_sequential_attack(y = y, t_start = t_start_s2,
                                     a_clean_all = a_clean, b_clean_all = b_clean,
                                     H = H_val, d_seq = rep(d_val, H_val),
                                     gamma_vec = gamma_vec, gamma_probs = gamma_probs,
                                     delta_vals = delta_vals_nonneg,
                                     wF = archetypes$FM$wF, wP = archetypes$FM$wP,
                                     cost_rate = cost_base, Kmax = Kmax_val, n_samples = n_samp_s2)
    cum_util_per_samp <- rowSums(res_tmp$util)
    util_H_list[[length(util_H_list) + 1]] <- data.frame(
      H = H_val, d = d_val,
      util_mean = mean(cum_util_per_samp),
      util_lo   = quantile(cum_util_per_samp, 0.025),
      util_hi   = quantile(cum_util_per_samp, 0.975))
  }
}
util_H_df <- bind_rows(util_H_list)
util_H_df$d_label <- factor(paste0("d = ", util_H_df$d), levels = paste0("d = ", d_grid_s24))

d0_df <- util_H_df %>% filter(d == 0) %>% arrange(H) %>%
  mutate(marginal = c(NA, diff(util_mean) / diff(H)))
initial_slope <- d0_df$marginal[!is.na(d0_df$marginal)][1]
dr_row        <- d0_df %>% filter(!is.na(marginal) & marginal <= 0.10 * initial_slope) %>% slice(1)
H_dr          <- if (nrow(dr_row) > 0) dr_row$H[1] else NA
util_at_dr    <- if (!is.na(H_dr)) dr_row$util_mean[1] else NA

p_s2_4 <- ggplot(util_H_df, aes(x = H, y = util_mean, color = d_label, fill = d_label)) +
  geom_ribbon(aes(ymin = util_lo, ymax = util_hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_color_viridis_d(option = "D", end = 0.85) +
  scale_fill_viridis_d(option  = "D", end = 0.85) +
  labs(title = "Cumulative Attacker Utility vs Horizon H (FM)",
       x = "Horizon H", y = "Cumulative Utility", color = NULL, fill = NULL) + theme_ara

if (!is.na(H_dr)) {
  p_s2_4 <- p_s2_4 +
    geom_vline(xintercept = H_dr, linetype = "dotted", color = "grey30", linewidth = 0.9) +
    annotate("text", x = H_dr + 0.4, y = util_at_dr * 0.60,
             label = paste0("H_dr = ", H_dr, "\n(<10% marginal\ngain vs initial)"),
             size = 3.0, hjust = 0, color = "grey30")
}

print(p_s2_4)
ggsave("S2_4_utility_vs_horizon.pdf", p_s2_4, width = 9, height = 5)

cat("\nAll S1 and S2 figures saved.\n")
cat("Outputs: S1_0_cost_sensitivity.pdf, S1_1_utility_profile.pdf,\n")
cat("         S1_2_distribution_shifts.pdf, S1_3_heatmap_threshold.pdf,\n")
cat("         S1_4_graybox_utility.pdf, S1_5_attack_success.pdf,\n")
cat("         S1_6_success_sensitivity.pdf, S2_1_filter_trajectory.pdf,\n")
cat("         S2_2_cumulative_KL.pdf, S2_4_utility_vs_horizon.pdf\n")