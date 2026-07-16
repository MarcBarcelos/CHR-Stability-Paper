# ================== STYLISTIC DISPERSION OVER TIME =================
# Core idea: Compute each text's absolute deviation from its fandom's median 
# factor score, then regress that deviation on time and other metadata with 
# plain ol vanilla OLS. 

# Coefficient `time` IS convergence/divergence answer 
#
# Brms is used for stage-2, pooling step (60 rows, fits in seconds) 
# ============================================================================

library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(ggplot2)

FACTORS <- paste0("F", 1:10)

# ==== 1. LOAD + PREP DATA ==========
load_and_prep <- function(csv_path = "data/final_full_dataset.csv") {
  d <- read.csv(csv_path)

  d$published     <- as.Date(d$published, "%Y-%m-%d")
  d$published_num <- as.numeric(d$published)
  # Normalizing dates so that they all are from 0-1 within fandom. This shifts 
  # focus to shape of distribution instead of time
  d <- d %>%
    group_by(fandom_label) %>%
    mutate(published_norm = (published_num - min(published_num, na.rm = TRUE)) /
             (max(published_num, na.rm = TRUE) - min(published_num, na.rm = TRUE))) %>%
    ungroup()

  # author-composition / community-growth proxies:
  # is_new_author - flags whether a text is that author's first-ever post in the
  # fandom, controlling for newcomers writing different to veterans
  d <- d %>%
    arrange(fandom_label, published) %>%
    group_by(fandom_label) %>%
    mutate(is_new_author = !duplicated(author),
           cum_n_authors = cumsum(is_new_author)) %>%
    ungroup()

  d <- d %>%
    mutate(rating = rating %>% str_remove_all("\\['|'\\]") %>% str_trim(),
           rating = factor(rating,
                           levels = c("General Audiences","Teen And Up Audiences",
                                      "Mature","Explicit","Not Rated")))

  d <- d %>%
    mutate(logwords        = as.numeric(scale(log(words))),
           time            = as.numeric(scale(published_norm)),
           log_cum_authors = as.numeric(scale(log(cum_n_authors))),
           fandom          = factor(fandom_label),
           author          = factor(author)) %>%
    filter(!is.na(time), !is.na(logwords), !is.na(rating), !is.na(log_cum_authors))

  # growth-rate residual: log_cum_authors with the time trend regressed out
  # per fandom, orthogonal to time by construction
  d <- d %>%
    group_by(fandom) %>%
    mutate(growth_resid = as.numeric(scale(residuals(lm(log_cum_authors ~ time))))) %>%
    ungroup()

  message("n works: ", nrow(d), " | authors: ", nlevels(d$author),
          " | fandoms: ", nlevels(d$fandom))
  d
}

d <- load_and_prep()

# ============================================================================
# 2. ABSOLUTE DEVIATION FROM MEDIAN, per fandom x factor
# Centering is done WITHIN FANDOM ONLY (not fandom x rating) so that rating
# can still be used as a predictor of dispersion below without being
# partialed out of the outcome itself.
# ============================================================================
d <- d %>%
  group_by(fandom) %>%
  mutate(across(all_of(FACTORS), ~ abs(.x - median(.x, na.rm = TRUE)),
                .names = "absdev_{.col}")) %>%
  ungroup()

# ============================================================================
# 3. OLS PER FANDOM x FACTOR (base R lm(), no MCMC)
# rating is dropped from the formula if any level has < 25 works in that
# fandom
# ============================================================================
tidy_lm <- function(m) {
  as.data.frame(coef(summary(m))) %>%
    tibble::rownames_to_column("term") %>%
    rename(estimate = Estimate, std.error = `Std. Error`,
           statistic = `t value`, p.value = `Pr(>|t|)`) %>%
    as_tibble()
}

fit_absdev_model <- function(d_f, factor_name, min_cell = 25) {
  y <- paste0("absdev_", factor_name)
  counts     <- d_f %>% count(rating, .drop = FALSE)
  use_rating <- all(counts$n >= min_cell) && nrow(counts) >= 2
  rhs <- if (use_rating) "time + logwords + rating + growth_resid + is_new_author"
         else            "time + logwords + growth_resid + is_new_author"
  f <- as.formula(paste(y, "~", rhs))
  lm(f, data = d_f)
}

absdev_results <- purrr::map_dfr(levels(d$fandom), function(fd) {
  d_f <- droplevels(filter(d, fandom == fd))
  purrr::map_dfr(FACTORS, function(fk) {
    m <- fit_absdev_model(d_f, fk)
    tidy_lm(m) %>% mutate(fandom = fd, factor = fk, .before = 1)
  })
})

# NOTE: absdev_results has 540 rows (6 fandoms x 10 factors x 9 terms) --
# deliberately NOT printed in full here. It's kept as an object so you can
# inspect/filter it directly (e.g. absdev_results %>% filter(fandom=="MAG")),
# but dumping all 540 rows to console just buries time_effects/shrunk below
# it. Use View(absdev_results) or filter it yourself if you want the raw table.

# ---- the main convergence/divergence answer: the `time` coefficient ----
time_effects <- absdev_results %>%
  filter(term == "time") %>%
  mutate(direction = case_when(
           estimate < 0 & p.value < 0.05 ~ "converging",
           estimate > 0 & p.value < 0.05 ~ "diverging",
           TRUE                          ~ "no credible change")) %>%
  select(fandom, factor, estimate, std.error, p.value, direction)

message("\n===== TIME EFFECT ON DISPERSION (converge/diverge), per fandom x factor =====")
print(time_effects, n = Inf)

# ---- metadata effects: does rating, length, growth, or new-author status
#      predict dispersion? (this is the "effect of other metadata" ask) ----
# NOTE: also not printed in full (~420 rows) for the same reason as
# absdev_results above -- filter it yourself, e.g.:
#   metadata_effects %>% filter(term == "growth_resid")
#   metadata_effects %>% filter(str_detect(term, "^rating"), p.value < 0.05)
metadata_effects <- absdev_results %>%
  filter(!term %in% c("(Intercept)", "time")) %>%
  arrange(fandom, factor, term)

# ============================================================================
# 4. OPTIONAL STAGE-2 POOLING — same shrinkage logic as convergence_fanfic.R,
# but now pooling the cheap OLS `time` coefficients instead of expensive
# brms sigma slopes. Still worth doing: you're still testing 60 cells, and
# partial pooling across fandom/factor still beats eyeballing 60 raw p-values.
# ============================================================================
suppressPackageStartupMessages(library(brms))

# `random` excluded from the pooling model: it's not a real fandom, so it
# shouldn't borrow statistical strength from (or lend strength to) the real
# fandoms' (1|fandom)/(1|factor) structure -- pooling it in previously turned
# its raw p=.085 F1 result (not even uncorrected-significant on its own) into
# a spurious "diverging" pooled result, purely from sharing the F1 factor-
# level tendency with HP/MAG/PJ. It's the intended internal-validity control,
# so it needs to stay genuinely independent of what's being pooled.
pool_data <- time_effects %>%
  filter(fandom != "random") %>%
  transmute(fandom = factor(fandom), factor = factor(factor), estimate, std.error)

pool_fit <- brm(
  estimate | se(std.error, sigma = TRUE) ~ 1 + (1 | fandom) + (1 | factor),
  data = pool_data, family = gaussian(),
  chains = 4, iter = 2000, cores = 4, seed = 1,
  backend = "cmdstanr", control = list(adapt_delta = 0.95),
  file = "outputs/brms_cache/brms_pool_absdev_shrinkage"
)

shrunk <- fitted(pool_fit, summary = TRUE) %>%
  as_tibble() %>%
  bind_cols(pool_data %>% select(fandom, factor, raw_estimate = estimate)) %>%
  rename(shrunk_estimate = Estimate, shrunk_lo95 = Q2.5, shrunk_hi95 = Q97.5) %>%
  mutate(direction = case_when(
           shrunk_hi95 < 0 ~ "converging",
           shrunk_lo95 > 0 ~ "diverging",
           TRUE             ~ "no credible change"),
         factor = as.character(factor)) %>%
  select(fandom, factor, raw_estimate, shrunk_estimate, shrunk_lo95, shrunk_hi95, direction)

message("\n===== STAGE-2 POOLED (shrunk) DISPERSION-TREND ESTIMATES (5 real fandoms) =====")
print(shrunk, n = Inf)

# random's own raw/uncorrected result, kept separate on purpose -- this is
# the actual control-group check: does the non-fandom baseline show anything
# at all, unpooled and un-shrunk?
message("\n===== random baseline (unpooled control, NOT part of shrinkage) =====")
print(time_effects %>% filter(fandom == "random"), n = Inf)

# ============================================================================
# 5. VISUAL SANITY CHECK: binned + smoothed dispersion over time.
# Point size = bin n, so you can see directly where estimates are backed by
# more or less data rather than treating every bin as equally trustworthy.
# ============================================================================
plot_dispersion_trend <- function(d, factor_name, n_bins = 15) {
  d %>%
    group_by(fandom) %>%
    mutate(time_bin = ntile(time, n_bins)) %>%
    group_by(fandom, time_bin) %>%
    summarise(time_mid    = mean(time),
              mean_absdev = mean(.data[[paste0("absdev_", factor_name)]]),
              n           = n(),
              .groups = "drop") %>%
    ggplot(aes(time_mid, mean_absdev)) +
    geom_point(aes(size = n), alpha = 0.5) +
    geom_smooth(method = "loess", se = TRUE, color = "#d7191c") +
    facet_wrap(~ fandom, scales = "free_y") +
    labs(title = paste("Dispersion over time (binned mean |deviation from median|) —", factor_name),
         x = "time (scaled, per-fandom)", y = "mean |deviation from fandom median|",
         size = "n texts\nin bin") +
    theme_bw(base_size = 11)
}

# usage:
# p <- plot_dispersion_trend(d, "F1"); print(p)
# ggsave("dispersion_trend_F1.png", p, width = 10, height = 6, dpi = 150)

# ---- headline heatmap: fandom x factor, colored by pooled slope, starred
# where credible. Direct analogue of convergence_fanfic.R's old heatmap, but
# built on the primary (OLS + shrinkage) pipeline's `shrunk` table. ----
plot_dispersion_heatmap <- function(shrunk) {
  shrunk %>%
    mutate(factor = factor(factor, levels = paste0("F", 1:10)),
           star   = ifelse(direction != "no credible change", "*", "")) %>%
    ggplot(aes(factor, fandom, fill = shrunk_estimate)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = star), size = 6, color = "black", vjust = 0.75) +
    scale_fill_gradient2(low = "#2c7bb6", mid = "grey95", high = "#d7191c",
                         midpoint = 0, name = "shrunk\nslope") +
    labs(title = "Stylistic dispersion trend by fandom and factor (pooled)",
         subtitle = "blue = converging, red = diverging; * = credible (95% CrI excludes 0)\nF4 = length proxy, F9 = length-adjusted",
         x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(face = "bold"))
}

# ---- raw vs pooled forest: shows the shrinkage itself, not just the result.
# Ordered by shrunk_estimate so the full converge/diverge profile reads at a
# glance; segment length shows how far pooling pulled each raw estimate. ----
plot_shrinkage_forest <- function(shrunk) {
  shrunk %>%
    mutate(cell = paste(fandom, factor, sep = " · "),
           cell = reorder(cell, shrunk_estimate)) %>%
    ggplot(aes(y = cell)) +
    geom_vline(xintercept = 0, linetype = 2, color = "grey50") +
    geom_segment(aes(x = raw_estimate, xend = shrunk_estimate, yend = cell),
                 color = "grey70", linewidth = 0.4) +
    geom_point(aes(x = raw_estimate), shape = 1, color = "grey60", size = 1.6) +
    geom_pointrange(aes(x = shrunk_estimate, xmin = shrunk_lo95, xmax = shrunk_hi95,
                         color = direction), size = 0.35) +
    scale_color_manual(values = c(converging = "#2c7bb6", diverging = "#d7191c",
                                   "no credible change" = "grey40"), name = NULL) +
    labs(title = "Dispersion-trend profile: raw OLS estimate vs pooled (shrunk) estimate",
         subtitle = "open grey circle = raw stage-1 slope; filled point + bar = pooled estimate (95% CrI)",
         x = "time slope", y = NULL) +
    theme_bw(base_size = 10) +
    theme(axis.text.y = element_text(size = 7))
}

# usage:
# p <- plot_dispersion_heatmap(shrunk); print(p)
# p <- plot_shrinkage_forest(shrunk); print(p)

# ============================================================================
# 6. COLLINEARITY CHECK (predictor set, per fandom) -- cheap sanity check.
# growth_resid is already orthogonalized to time by construction, so this
# should come back clean (~1), but confirm before trusting metadata_effects.
# ============================================================================
compute_vif <- function(d_f, predictors = c("time","logwords","growth_resid","is_new_author")) {
  purrr::map_dfr(predictors, function(p) {
    other <- setdiff(predictors, p)
    f  <- as.formula(paste(p, "~", paste(other, collapse = " + ")))
    r2 <- summary(lm(f, data = d_f))$r.squared
    tibble(term = p, vif = 1 / (1 - r2))
  })
}

vif_by_fandom <- d %>%
  split(.$fandom) %>%
  purrr::map_dfr(function(d_f) compute_vif(d_f) %>% mutate(fandom = unique(d_f$fandom), .before = 1))

message("\n===== VIF: absolute-deviation model predictors, per fandom (VIF > 5 = concerning) =====")
print(vif_by_fandom, n = Inf)

# ============================================================================
# 7. RATING x TIME INTERACTION — does the RATE of convergence/divergence
# differ by content rating? (analogue of Model 2's `sigma ~ rating*time` in
# convergence_fanfic.R.) fit_absdev_model() above only has rating as an
# additive term (does rating shift the average dispersion level); this adds
# rating*time so the `time` slope itself is allowed to vary by rating.
#
# Countermeasures applied here (see concerns raised about this analysis):
#   1. min_cell raised to 100 (vs 25 for the additive model) -- an
#      interaction slope needs far more data than a level-shift to be
#      estimated reliably.
#   2. "Not Rated" excluded from the interaction test -- it's a metadata-
#      quality bucket (rating left blank), not a coherent substantive
#      category, so a "trend" there is hard to interpret.
#   3. n_works per fandom x rating reported alongside every estimate, so
#      precision is visible directly instead of trusting a bare p-value.
#   4. rating_time_collinearity checks whether a fandom's rating mix shifted
#      over its own timeline -- the same failure mode growth_resid was built
#      to fix for community growth vs time (VIF ~50 in RPF/HP, see earlier).
#   5. total_slope = baseline `time` + interaction, so you read a rating's
#      actual trend instead of just its shift from the reference level.
#   6. Stage-2 pooling (shrunk_rt) replaces raw uncorrected p<.05 as the
#      reported criterion, same partial-pooling logic as sections 4 and 8.
# ============================================================================
fit_absdev_model_ratingtime <- function(d_f, factor_name, min_cell = 100) {
  y <- paste0("absdev_", factor_name)
  d_f <- d_f %>% filter(rating != "Not Rated") %>% droplevels()
  counts     <- d_f %>% count(rating, .drop = FALSE)
  use_rating <- all(counts$n >= min_cell) && nrow(counts) >= 2
  if (!use_rating) return(NULL)  # too thin to test rating*time for this fandom
  f <- as.formula(paste(y, "~ time * rating + logwords + growth_resid + is_new_author"))
  lm(f, data = d_f)
}

# ---- collinearity check: is rating confounded with time within each fandom? ----
rating_time_collinearity <- purrr::map_dfr(levels(d$fandom), function(fd) {
  d_f <- d %>% filter(fandom == fd, rating != "Not Rated") %>% droplevels()
  r2_full  <- summary(lm(time ~ rating + logwords + growth_resid + is_new_author, data = d_f))$r.squared
  r2_norat <- summary(lm(time ~           logwords + growth_resid + is_new_author, data = d_f))$r.squared
  tibble(fandom = fd, incremental_r2_from_rating = r2_full - r2_norat)
})

message("\n===== COLLINEARITY: how much rating alone explains time, per fandom =====")
message("(> ~0.10 -- rating mix shifted meaningfully over this fandom's timeline; discount its interaction results)")
print(rating_time_collinearity %>% arrange(desc(incremental_r2_from_rating)), n = Inf)

absdev_results_rt <- purrr::map_dfr(levels(d$fandom), function(fd) {
  d_f <- droplevels(filter(d, fandom == fd))
  purrr::map_dfr(FACTORS, function(fk) {
    m <- fit_absdev_model_ratingtime(d_f, fk)
    if (is.null(m)) return(NULL)
    tidy_lm(m) %>% mutate(fandom = fd, factor = fk, .before = 1)
  })
})

# baseline `time` slope per fandom x factor (General Audiences' own trend)
baseline_time <- absdev_results_rt %>%
  filter(term == "time") %>%
  select(fandom, factor, baseline_time_estimate = estimate)

# per-fandom x rating sample size, for reading precision directly
rating_n <- d %>%
  filter(rating != "Not Rated") %>%
  count(fandom, rating, name = "n_works")

# the interaction terms: does the time slope differ by rating, relative to
# General Audiences? total_slope = baseline_time_estimate + estimate is the
# rating's own actual trend, not just its shift from the reference level.
rating_time_effects <- absdev_results_rt %>%
  filter(str_detect(term, "^time:rating")) %>%
  mutate(rating = str_remove(term, "^time:rating")) %>%
  left_join(baseline_time, by = c("fandom", "factor")) %>%
  mutate(total_slope = baseline_time_estimate + estimate) %>%
  left_join(rating_n, by = c("fandom", "rating")) %>%
  select(fandom, factor, rating, n_works, estimate, std.error, p.value,
         baseline_time_estimate, total_slope) %>%
  arrange(fandom, factor, rating)

message("\n===== RATING x TIME interaction on dispersion (uncorrected p<.05 only) =====")
print(rating_time_effects %>% filter(p.value < 0.05), n = Inf)
message("(full table in `rating_time_effects`, ", nrow(rating_time_effects),
        " rows, uncorrected -- filter/View as needed)")

# ---- stage-2 pooling for the interaction estimates (replaces raw p<.05 as
# the reported criterion) -- same shrinkage logic as sections 4 and 8 ----
pool_data_rt <- rating_time_effects %>%
  filter(!is.na(std.error), fandom != "random") %>%  # same reasoning as section 4
  transmute(fandom = factor(fandom), factor = factor(factor), rating = factor(rating),
            estimate, std.error)

# (1 | rating) replaced with a FIXED `rating` term, not dropped entirely --
# removing it outright (previous attempt) left the model with no way to
# distinguish which rating produced a given estimate, so it silently
# collapsed all 3 ratings per fandom x factor to one shared pooled value
# (visible as identical shrunk_estimate/CI across ratings -- a bug, not a
# result). A fixed effect only estimates 2 extra parameters (vs. a variance
# component over 3 groups), preserves per-rating distinctness, and should
# still sample cleanly since it's not asking for a poorly-identified
# hierarchical variance from only 3 levels.
pool_fit_rt <- brm(
  estimate | se(std.error, sigma = TRUE) ~ 1 + rating + (1 | fandom) + (1 | factor),
  data = pool_data_rt, family = gaussian(),
  chains = 4, iter = 2000, cores = 4, seed = 1,
  backend = "cmdstanr", control = list(adapt_delta = 0.99),
  file = "outputs/brms_cache/brms_pool_ratingtime_shrinkage"
)

shrunk_rt <- fitted(pool_fit_rt, summary = TRUE) %>%
  as_tibble() %>%
  bind_cols(pool_data_rt %>% select(fandom, factor, rating, raw_estimate = estimate)) %>%
  rename(shrunk_estimate = Estimate, shrunk_lo95 = Q2.5, shrunk_hi95 = Q97.5) %>%
  mutate(shift_direction = case_when(
           shrunk_hi95 < 0 ~ "shifts toward convergence",
           shrunk_lo95 > 0 ~ "shifts toward divergence",
           TRUE             ~ "no credible shift")) %>%
  select(fandom, factor, rating, raw_estimate, shrunk_estimate, shrunk_lo95, shrunk_hi95, shift_direction)

message("\n===== STAGE-2 POOLED (shrunk) RATING x TIME INTERACTION (5 real fandoms) =====")
print(shrunk_rt, n = Inf)

message("\n===== random baseline (unpooled control, NOT part of shrinkage) =====")
print(rating_time_effects %>% filter(fandom == "random", p.value < 0.05), n = Inf)

# ---- heatmap: fandom x factor, faceted by rating, colored by the pooled
# rating:time interaction -- does this rating shift the convergence/
# divergence RATE, and in which fandoms? ----
plot_rating_time_heatmap <- function(shrunk_rt) {
  shrunk_rt %>%
    mutate(factor = factor(factor, levels = paste0("F", 1:10)),
           star   = ifelse(shift_direction != "no credible shift", "*", "")) %>%
    ggplot(aes(factor, fandom, fill = shrunk_estimate)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = star), size = 5, color = "black", vjust = 0.75) +
    scale_fill_gradient2(low = "#2c7bb6", mid = "grey95", high = "#d7191c",
                         midpoint = 0, name = "shrunk\nshift") +
    facet_wrap(~ rating) +
    labs(title = "Rating's effect on the convergence/divergence RATE, relative to General Audiences",
         subtitle = "red = shifts toward divergence, blue = shifts toward convergence; * = credible",
         x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(face = "bold"))
}

# usage:
# p <- plot_rating_time_heatmap(shrunk_rt); print(p)

# ============================================================================
# 8. RATING x TIME EFFECT, COMPARED ACROSS FANDOMS
# Does rating change the convergence/divergence RATE (not just the average
# dispersion level) differently from fandom to fandom? rating_time_effects
# above already has fandom as a column, so no new model is needed -- this
# just summarizes/visualizes it for cross-fandom comparison instead of
# reading it as one long table.
# ============================================================================
# for each rating x factor, how many fandoms show a credible shift in the
# time-trend? a rating that's credible in 1/6 fandoms is fandom-specific; one
# that's credible in most/all fandoms is more likely a general rating effect.
rating_time_summary <- rating_time_effects %>%
  group_by(rating, factor) %>%
  summarise(n_fandoms_tested = n(),
            n_credible       = sum(p.value < 0.05),
            fandoms_credible = paste(fandom[p.value < 0.05], collapse = ", "),
            .groups = "drop") %>%
  arrange(desc(n_credible))

message("\n===== RATING x TIME: how many fandoms show a credible shift, per rating x factor =====")
print(rating_time_summary %>% filter(n_credible > 0), n = Inf)

# visual: for one factor at a time, does rating's effect on the convergence/
# divergence RATE differ by fandom?
plot_rating_time_by_fandom <- function(rating_time_effects, factor_name) {
  rating_time_effects %>%
    filter(factor == factor_name) %>%
    ggplot(aes(x = rating, y = estimate, color = fandom)) +
    geom_hline(yintercept = 0, linetype = 2, color = "grey50") +
    geom_pointrange(aes(ymin = estimate - 1.96 * std.error, ymax = estimate + 1.96 * std.error),
                     position = position_dodge(width = 0.5)) +
    labs(title = paste("Does rating shift the convergence/divergence rate, by fandom —", factor_name),
         subtitle = "time x rating interaction, relative to General Audiences",
         x = "rating", y = "shift in time-slope (dispersion trend)") +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

# usage:
# p <- plot_rating_time_by_fandom(rating_time_effects, "F1"); print(p)

# ============================================================================
# 9. ADDITIVE METADATA EFFECTS — does rating shift the AVERAGE dispersion
# level (not just its rate of change, that's section 7/8), and what about
# logwords/growth_resid/is_new_author? These have been sitting in
# metadata_effects since section 3 but never actually surfaced.
# ============================================================================

# ---- 9a. rating's average-level effect, pooled the same way as sections 4/7 ----
rating_level_raw <- metadata_effects %>%
  filter(str_detect(term, "^rating")) %>%
  mutate(rating = str_remove(term, "^rating")) %>%
  filter(fandom != "random")  # same reasoning as sections 4/7

pool_data_rl <- rating_level_raw %>%
  transmute(fandom = factor(fandom), factor = factor(factor), rating = factor(rating),
            estimate, std.error)

pool_fit_rl <- brm(
  estimate | se(std.error, sigma = TRUE) ~ 1 + rating + (1 | fandom) + (1 | factor),
  data = pool_data_rl, family = gaussian(),
  chains = 4, iter = 2000, cores = 4, seed = 1,
  backend = "cmdstanr", control = list(adapt_delta = 0.99),
  file = "outputs/brms_cache/brms_pool_ratinglevel_shrinkage"
)

shrunk_rl <- fitted(pool_fit_rl, summary = TRUE) %>%
  as_tibble() %>%
  bind_cols(pool_data_rl %>% select(fandom, factor, rating, raw_estimate = estimate)) %>%
  rename(shrunk_estimate = Estimate, shrunk_lo95 = Q2.5, shrunk_hi95 = Q97.5) %>%
  mutate(level_direction = case_when(
           shrunk_hi95 < 0 ~ "lower dispersion than General Audiences",
           shrunk_lo95 > 0 ~ "higher dispersion than General Audiences",
           TRUE             ~ "no credible difference")) %>%
  select(fandom, factor, rating, raw_estimate, shrunk_estimate, shrunk_lo95, shrunk_hi95, level_direction)

message("\n===== STAGE-2 POOLED: rating's effect on AVERAGE dispersion level (5 real fandoms) =====")
print(shrunk_rl %>% filter(level_direction != "no credible difference"), n = Inf)
message("(full table in `shrunk_rl`, ", nrow(shrunk_rl), " rows)")

# ---- heatmap: fandom x factor, faceted by rating, colored by the pooled
# rating LEVEL effect -- this is the one that visualizes the F1/F9/F10 vs
# F3/F4/F7/F8 split (higher vs lower dispersion in rated content). ----
plot_rating_level_heatmap <- function(shrunk_rl) {
  shrunk_rl %>%
    mutate(factor = factor(factor, levels = paste0("F", 1:10)),
           star   = ifelse(level_direction != "no credible difference", "*", "")) %>%
    ggplot(aes(factor, fandom, fill = shrunk_estimate)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = star), size = 5, color = "black", vjust = 0.75) +
    scale_fill_gradient2(low = "#2c7bb6", mid = "grey95", high = "#d7191c",
                         midpoint = 0, name = "shrunk\nestimate") +
    facet_wrap(~ rating) +
    labs(title = "Rating's effect on AVERAGE dispersion level, relative to General Audiences",
         subtitle = "red = higher dispersion than G-rated content, blue = lower; * = credible",
         x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(face = "bold"))
}

# usage:
# p <- plot_rating_level_heatmap(shrunk_rl); print(p)

# ---- 9b. logwords / growth_resid / is_new_author -- quick visibility pass ----
# not pooled (these are nuisance covariates, not a primary question here),
# but at least surfaced: per term x fandom, how many factors show p<.05,
# and which direction the effect leans on average.
other_metadata_summary <- metadata_effects %>%
  filter(term %in% c("logwords", "growth_resid", "is_new_authorTRUE")) %>%
  group_by(term, fandom) %>%
  summarise(n_factors_tested = n(),
            n_credible        = sum(p.value < 0.05),
            mean_estimate     = mean(estimate),
            .groups = "drop") %>%
  arrange(term, desc(n_credible))

message("\n===== logwords / growth_resid / is_new_author: credible-count summary, per term x fandom =====")
print(other_metadata_summary, n = Inf)
message("(row-level detail in `metadata_effects %>% filter(term == \"logwords\")` etc.)")

# ============================================================================
# 10. NON-LINEARITY ROBUSTNESS CHECK
# Motivated by actually looking at plot_dispersion_trend(): several credible
# cells (MAG/PJ F8 especially) show a clear inverted-U over time, not a
# monotonic trend -- the linear `time` term averages the rise and the recent
# decline into one slope, which can still come out "diverging" even though
# "rose then partially reversed" is the more accurate story. For every
# credible cell in `shrunk` (5 real fandoms), refit with a flexible
# (natural spline) time term and check (a) whether the non-linear terms
# significantly improve on the linear fit, and (b) whether the fitted
# early-vs-late direction still agrees with the linear model's label.
# ============================================================================
fit_absdev_model_spline <- function(d_f, factor_name, min_cell = 25) {
  y <- paste0("absdev_", factor_name)
  counts     <- d_f %>% count(rating, .drop = FALSE)
  use_rating <- all(counts$n >= min_cell) && nrow(counts) >= 2
  rhs_linear <- if (use_rating) "time + logwords + rating + growth_resid + is_new_author"
                else            "time + logwords + growth_resid + is_new_author"
  rhs_spline <- sub("^time", "splines::ns(time, df = 3)", rhs_linear)
  list(
    linear = lm(as.formula(paste(y, "~", rhs_linear)), data = d_f),
    spline = lm(as.formula(paste(y, "~", rhs_spline)), data = d_f)
  )
}

credible_cells <- shrunk %>%
  filter(direction != "no credible change") %>%
  mutate(fandom = as.character(fandom)) %>%
  select(fandom, factor, direction)

nonlinearity_check <- purrr::pmap_dfr(credible_cells, function(fandom, factor, direction) {
  d_f <- droplevels(filter(d, as.character(.data$fandom) == !!fandom))
  m   <- fit_absdev_model_spline(d_f, factor)

  # does the non-linear term significantly improve on the linear fit?
  nonlin_p <- anova(m$linear, m$spline)[["Pr(>F)"]][2]

  # fitted trend shape at early/mid/late time, holding other covariates at
  # representative values (rating ignored by predict() if not in the formula)
  newdata <- tibble(
    time          = quantile(d_f$time, probs = c(0.05, 0.5, 0.95)),
    logwords      = mean(d_f$logwords),
    growth_resid  = mean(d_f$growth_resid),
    is_new_author = FALSE,
    rating        = "General Audiences"
  )
  pred <- predict(m$spline, newdata = newdata)

  tibble(fandom = fandom, factor = factor, linear_direction = direction,
         nonlinear_p = nonlin_p,
         fitted_early = pred[1], fitted_mid = pred[2], fitted_late = pred[3],
         net_change = pred[3] - pred[1],
         shape_agrees_with_linear = sign(net_change) == ifelse(direction == "diverging", 1, -1))
})

message("\n===== NON-LINEARITY CHECK: do credible linear findings survive a flexible time term? =====")
message("(nonlinear_p < .05 = the curve is significantly non-linear;")
message(" shape_agrees_with_linear = FALSE = the early-vs-late direction contradicts the linear label)")
print(nonlinearity_check %>% arrange(nonlinear_p), n = Inf)

# ---- visual: linear fit vs flexible (spline) fit, overlaid on the same
# binned data as plot_dispersion_trend(), for one fandom x factor at a time ----
plot_spline_comparison <- function(d, fandom_name, factor_name, n_bins = 15) {
  d_f <- d %>% filter(fandom == fandom_name)
  y_col <- paste0("absdev_", factor_name)

  binned <- d_f %>%
    mutate(time_bin = ntile(time, n_bins)) %>%
    group_by(time_bin) %>%
    summarise(time_mid = mean(time), mean_absdev = mean(.data[[y_col]]), n = n(), .groups = "drop")

  ggplot(binned, aes(time_mid, mean_absdev)) +
    geom_point(aes(size = n), alpha = 0.5) +
    geom_smooth(method = "lm", se = FALSE, color = "#2c7bb6", linewidth = 0.8) +
    geom_smooth(method = "lm", formula = y ~ splines::ns(x, df = 3), se = TRUE,
                color = "#d7191c", linewidth = 0.8) +
    labs(title = paste(fandom_name, "—", factor_name, ": linear (blue) vs flexible (red) time fit"),
         subtitle = "binned mean |deviation from median|; red ribbon = spline 95% CI",
         x = "time (scaled)", y = "mean |deviation from fandom median|", size = "n texts\nin bin") +
    theme_bw(base_size = 11)
}

# usage:
# p <- plot_spline_comparison(d, "MAG", "F8"); print(p)
