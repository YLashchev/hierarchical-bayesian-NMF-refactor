# plot_utilities.R
# This version works with the merged CSV that already contains model outputs
# Adapted from dobbs_fertility/plot_utilities.R to work with new data structure

library(tidybayes)
library(kableExtra)
library(ggrepel)
library(patchwork)
library(gt)
library(tidyverse)
library(lubridate)

# NOTE: merge_draws_and_data is NO LONGER NEEDED!
# The data is already merged in the CSV files from the Python pipeline
# The merged data has columns: .draw, .chain, K, D, N, category, state, time, 
# births, population, ypred, mu, mu_treated, exposure_code, banned_state, etc.

#' Make combined fit and gap plots with histogram
#'
#' @param merged_df Data frame with merged draws and data
#' @param quantiles_df Data frame with quantiles computed from merged_df
#' @param state_name Name of the state to plot
#' @param category Category to filter (default: "Total")
#' @param target Target variable name (default: "births")
#' @return A patchwork plot combining fit plot, gap plot, and histogram
#' 
make_all_te_plots <- function(merged_df, quantiles_df, state_name = "Texas", 
                               category = "Total", target = "births") {
  p1 <- make_state_fit_plot(quantiles_df, state_name, category = category, target = target)
  p2 <- make_gap_plot(quantiles_df, state_name, target = target, category = category)
  p3 <- make_births_histogram(merged_df, state_name = state_name, category = category, target = target)
  
  (p1 + p2) / p3
}

#' Make state fit plot showing observed vs predicted
#'
#' @param quantiles_df Data frame with quantiles (ypred_mean, ypred_lower, ypred_upper)
#' @param state_name Name of the state to plot
#' @param category Category to filter (default: "Total")
#' @param target Target variable name (default: "births")
#' @return A ggplot object
make_state_fit_plot <- function(quantiles_df, state_name, category = "Total", target = "births") {
  
  quantiles_df <- quantiles_df %>% filter(category == !!category)
  
  state_plot <- quantiles_df %>% 
    filter(state == !!state_name) %>% 
    ggplot() + 
    geom_point(aes(x = time, y = .data[[target]])) + 
    geom_ribbon(aes(x = time, ymin = ypred_lower, ymax = ypred_upper), alpha = 0.5) + 
    geom_line(aes(x = time, y = ypred_mean), color = "red") + 
    theme_bw() + 
    xlab("Date") + 
    ggtitle(state_name)
  
  if (target == "births") {
    state_plot <- state_plot + ylab("Births")
  } else {
    state_plot <- state_plot + ylab("Deaths")
  }
  
  # Check if this is a banned state and add treatment line
  banned_check <- quantiles_df %>% 
    filter(state == !!state_name) %>% 
    pull(banned_state) %>% 
    first()
  
  if (!is.na(banned_check) && banned_check == 1) {
    treatment_date <- quantiles_df %>% 
      filter(state == !!state_name, exposure_code == 1) %>% 
      summarize(treatment_date = first(time)) %>% 
      pull(treatment_date)
    
    if (length(treatment_date) > 0 && !is.na(treatment_date)) {
      state_plot <- state_plot + geom_vline(xintercept = treatment_date, linetype = "dashed")
    }
  }
  
  state_plot
}

#' Make gap plot showing difference between observed and predicted
#'
#' @param quantiles_df Data frame with quantiles
#' @param state_name Name of the state to plot (default: "Texas")
#' @param category Category to filter (default: "Total")
#' @param target Target variable name (default: "births")
#' @return A ggplot object
make_gap_plot <- function(quantiles_df, state_name = "Texas", category = "Total", target = "births") {
  
  quantiles_df <- quantiles_df %>% filter(category == !!category & state == !!state_name)
  
  # Check if this is a banned state
  banned_check <- quantiles_df %>% 
    pull(banned_state) %>% 
    first()
  
  if (!is.na(banned_check) && banned_check == 1) {
    treatment_date <- quantiles_df %>%
      filter(exposure_code == 1) %>%
      summarize(treatment_date = first(time)) %>%
      pull(treatment_date)
    if (length(treatment_date) == 0) treatment_date <- NA
  } else {
    treatment_date <- NA
  }
  
  quantiles_df %>% 
    mutate(pre_ban = time < treatment_date) %>%
    ggplot() +
    geom_ribbon(aes(
      x = time,
      ymax = .data[[target]] / ypred_lower - 1,
      ymin = .data[[target]] / ypred_upper - 1,
      group = pre_ban
    ), alpha = 0.25) +
    geom_line(aes(x = time, y = .data[[target]] / ypred_mean - 1, group = pre_ban), color = "red") +
    theme_bw() +
    geom_hline(yintercept = 0, col = "black", linetype = "dashed", alpha = 0.75) +
    xlab("Date") -> gap_plot
  
  if (target == "births") {
    gap_plot <- gap_plot + ylab("Observed / Predicted Births - 1")
  } else {
    gap_plot <- gap_plot + ylab("Observed / Predicted Deaths - 1")
  }
  
  if (!is.na(treatment_date)) {
    gap_plot <- gap_plot + geom_vline(xintercept = treatment_date, linetype = "dashed")
  }
  
  gap_plot
}

#' Make histogram of total births difference (observed - predicted)
#'
#' @param merged_df Data frame with merged draws and data
#' @param state_name Name of the state (default: "Texas")
#' @param category Category to filter (default: "Total")
#' @param treatment_date Optional treatment date (will be inferred if NULL)
#' @param target Target variable name (default: "births")
#' @return A ggplot object
make_births_histogram <- function(merged_df, state_name = "Texas", category = "Total", 
                                  treatment_date = NULL, target = "births") {
  
  merged_df <- merged_df %>%
    filter(
      category == !!category,
      state == !!state_name
    )
  
  if (is.null(treatment_date)) {
    merged_df %>% filter(exposure_code == 1) %>% 
      summarize(treatment_date = first(time)) %>% 
      pull(treatment_date) -> treatment_date
  }
  
  totals_df <- merged_df %>%
    filter(time >= treatment_date) %>%
    group_by(.draw) %>%
    summarize(obs = sum(.data[[target]]), pred = sum(ypred)) %>%
    mutate(diff = obs - pred)
  
  pval <- round(mean(totals_df$pred > totals_df$obs), 2)
  
  hist_plot <- totals_df %>% ggplot() + 
    geom_histogram(aes(x = diff), bins = 50) + 
    geom_vline(xintercept = 0, col = "red", linetype = "dashed") + 
    theme_bw() 
  
  if (target == "births") {
    hist_plot <- hist_plot + 
      ggtitle("Difference in Observed and Predicted Total Births") +
      xlab("Births")
  } else {
    hist_plot <- hist_plot + 
      ggtitle("Difference in Observed and Predicted Total Deaths") +
      xlab("Deaths")
  }
  
  hist_plot + annotate("text",
                       x = Inf, y = Inf,
                       label = sprintf("Pval = %.2f", pval),
                       hjust = 1.2, vjust = 1.8
  )
}

#' Generate violin plots showing causal effects by group
#'
#' @param merged_df Data frame with merged draws and data
#' @param states Optional vector of states to include
#' @param treatment_date Optional treatment date
#' @param group_var Grouping variable (default: "state")
#' @param categories Optional vector of categories to include
#' @param target Target variable (default: "births")
#' @param denom Denominator variable (default: "population")
#' @param rate_normalizer Rate normalizer (default: 1000)
#' @param estimand Estimand type: "diff" or "ratio" (default: "diff")
#' @param method Method: "pred" or "mu" (default: "pred")
#' @return A ggplot object
make_violins <- function(merged_df, states = NULL, treatment_date = NULL,
                         group_var = "state",
                         categories = NULL, target = "births", denom = "population",
                         rate_normalizer = 1000,
                         estimand = "diff", 
                         method = "pred") {
  
  if (is.null(states)) { 
    states <- unique(merged_df$state[merged_df$banned_state == 1])
  }
  
  if (is.null(categories)) {
    categories <- unique(merged_df$category)
  } else {
    merged_df <- merged_df %>% filter(category %in% categories) 
  } 
  
  merged_df <- merged_df %>%
    filter(exposure_code == 1) %>%
    mutate(years = interval(start_date, end_date) / years(1))
  
  # Remove Ban States aggregates, will be recomputed below
  merged_df <- merged_df %>% filter(!state %in% c("Ban States", "Ban States (excl. Texas)"))
  
  if (method == "pred") {
    if (target == "births") {
      # Compute ratio of birth rates (rates per 1000 people per year)
      state_df <- merged_df %>%
        group_by_at(c(".draw", group_var, "time")) %>%
        mutate(outcome = sum(.data[[target]]), ypred = sum(ypred), denom_val = mean(.data[[denom]])) %>%
        mutate(
          outcome_rate = (outcome / years) / (denom_val / rate_normalizer),
          ypred_rate = ypred / years / (denom_val / rate_normalizer)
        ) %>%
        ungroup() %>%
        group_by_at(c(".draw", group_var)) %>%
        summarize(
          causal_effect_diff = mean(outcome_rate - ypred_rate),
          causal_effect_ratio = mean(outcome_rate / ypred_rate),
          causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio),
          .groups = "drop"
        ) %>%
        ungroup()
    } else {
      # Compute difference in death rate per 1000 births
      state_df <- merged_df %>%
        group_by_at(c(".draw", group_var, "time")) %>%
        mutate(outcome = sum(.data[[target]]), ypred = sum(ypred), denom_val = sum(.data[[denom]])) %>%
        mutate(
          outcome_rate = outcome / (denom_val / rate_normalizer),
          ypred_rate = ypred / (denom_val / rate_normalizer)
        ) %>%
        ungroup() %>%
        group_by_at(c(".draw", group_var)) %>%
        mutate(
          causal_effect_ratio = mean(outcome_rate / ypred_rate),
          causal_effect_diff = mean(outcome_rate - ypred_rate),
          causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio)
        ) %>%
        ungroup()
    }
  } else {
    if (target == "births") {
      # Compute using mu and mu_treated
      state_df <- merged_df %>%
        group_by_at(c(".draw", group_var, "time", "state")) %>%
        mutate(treated = sum(exp(mu_treated)), untreated = sum(exp(mu)), denom_val = mean(.data[[denom]] * years)) %>%
        ungroup()
      
      ban_states_df <- state_df %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = max(state_df$D) + 1, state = "Ban States", banned_state = TRUE)
      
      ban_states_no_tx_df <- state_df %>%
        filter(state != "Texas" & state != "TX") %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = max(state_df$D) + 2, state = "Ban States (excl. Texas)", banned_state = TRUE)
      
      state_df <- bind_rows(state_df %>% select_at(colnames(ban_states_df)), ban_states_df, ban_states_no_tx_df)
      state_df <- state_df %>% filter(state %in% states)
      
      state_df <- state_df %>% group_by_at(c(".draw", group_var)) %>%
        summarize(
          treated_rate = sum(treated) / sum(denom_val) * rate_normalizer,
          untreated_rate = sum(untreated) / sum(denom_val) * rate_normalizer,
          causal_effect_diff = treated_rate - untreated_rate,
          causal_effect_ratio = treated_rate / untreated_rate,
          causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio),
          .groups = "drop"
        ) %>% ungroup()
      
    } else {
      # Deaths computation
      state_df <- merged_df %>%
        group_by_at(c(".draw", group_var, "time")) %>%
        mutate(treated = sum(exp(mu_treated)), untreated = sum(exp(mu)), denom_val = sum(.data[[denom]])) %>%
        ungroup()
      
      ban_states_df <- state_df %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = max(state_df$D) + 1, state = "Ban States", banned_state = TRUE)
      
      ban_states_no_tx_df <- state_df %>%
        filter(state != "Texas" & state != "TX") %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = max(state_df$D) + 2, state = "Ban States (excl. Texas)", banned_state = TRUE)
      
      state_df <- bind_rows(state_df %>% select_at(colnames(ban_states_df)), ban_states_df, ban_states_no_tx_df)
      state_df <- state_df %>% filter(state %in% states)
      
      state_df <- state_df %>% group_by_at(c(".draw", group_var)) %>%
        summarize(
          treated_rate = sum(treated) / sum(denom_val) * rate_normalizer,
          untreated_rate = sum(untreated) / sum(denom_val) * rate_normalizer,
          causal_effect_diff = treated_rate - untreated_rate,
          causal_effect_ratio = treated_rate / untreated_rate,
          causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio),
          .groups = "drop"
        ) %>% ungroup()
    }
  }
  
  state_df <- state_df %>%    
    mutate({{group_var}} := factor(.data[[group_var]])) %>%
    mutate({{group_var}} := fct_reorder(.data[[group_var]], causal_effect, .fun = median)) 
  
  stats_df <- state_df %>%
    group_by_at(group_var) %>%
    summarize(
      mean = round(mean(causal_effect), 3),
      pval = if (method == "pred") { 
        round(2 * mean(untreated_rate > treated_rate), 3) 
      } else { 
        round(mean(treated_rate <= untreated_rate), 3) 
      },
      Significance = factor(ifelse(pval < 0.05, "red", "black"), levels = c("red", "black")),
      .groups = "drop"
    ) %>%
    ungroup()
  
  vp <- state_df %>%
    ggplot(aes(x = .data[[group_var]], y = causal_effect)) +
    geom_violin(fill = "gray", alpha = 0.5, draw_quantiles = c(0.5)) +
    geom_hline(yintercept = ifelse(estimand == "diff", 0, 1), col = "red", linetype = "dashed") +
    geom_text(data = stats_df, aes(x = .data[[group_var]], y = Inf, label = pval, col = Significance), 
              vjust = 2, fontface = "bold") +
    scale_colour_manual(values = c("red" = "red", "black" = "black")) +
    guides(colour = "none") +
    geom_text(data = stats_df, aes(x = .data[[group_var]], y = Inf, label = mean), vjust = 4, fontface = "bold") +
    theme_bw(base_size = 16) +    
    xlab(group_var) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  
  if (target == "births") {
    vp <- vp + labs(title = ifelse(estimand == "diff", "Post. Pred. Difference in fertility rate", 
                                    "Posterior Predictive Mult. change in fertility rate"))
    vp <- vp + ylab("Causal Effect")
  } else {
    vp <- vp + labs(title = ifelse(estimand == "diff", "Post. Pred. Difference in mortality rate", 
                                    "Post. Pred. Mult. change in mortality rate"))
    vp <- vp + ylab("Mult. Change in deaths (per 1k live births)")
  }
  vp
}

#' Generate interval plots showing causal effects with credible intervals
#'
#' @param merged_df Data frame with merged draws and data
#' @param states Optional vector of states to include
#' @param group_var Grouping variable (default: "state")
#' @param categories Optional vector of categories to include
#' @param target Target variable (default: "births")
#' @param denom Denominator variable (default: "population")
#' @param rate_normalizer Rate normalizer (default: 1000)
#' @param estimand Estimand type: "diff" or "ratio" (default: "diff")
#' @param method Method: "pred" or "mu" (default: "mu")
#' @param x_var Variable for x-axis (default: "state")
#' @param color_group Variable for color grouping
#' @return A ggplot object
make_interval_plot <- function(merged_df, 
                               states = NULL,
                               group_var = "state",
                               categories = NULL, 
                               target = "births", 
                               denom = "population",
                               rate_normalizer = 1000,
                               estimand = "diff", 
                               method = "mu",
                               x_var = "state",
                               color_group = setdiff(group_var, "state")) {
  
  if (is.null(categories)) {
    categories <- unique(merged_df$category)
  } else {
    merged_df <- merged_df %>% filter(category %in% categories) 
  } 
  
  merged_df <- merged_df %>%
    filter(exposure_code == 1) %>%
    mutate(years = interval(start_date, end_date) / years(1))
  
  # Check if Ban States aggregates already exist in the data
  has_ban_aggregates <- any(merged_df$state %in% c("Ban States", "Ban States (excl. Texas)"))
  
  # Set default states BEFORE filtering out aggregates
  if (is.null(states)) { 
    states <- unique(merged_df$state[merged_df$banned_state == 1])
  }
  
  # Only remove and recompute if they DON'T exist
  if (!has_ban_aggregates) {
    create_aggregates <- TRUE
  } else {
    create_aggregates <- FALSE
    merged_df_orig <- merged_df  # Keep original with aggregates
    merged_df <- merged_df %>% filter(!state %in% c("Ban States", "Ban States (excl. Texas)"))
  }
  
  if (method == "mu") {
    if (target == "births") {
      # Compute using mu and mu_treated - EXACTLY like dobbs_fertility
      state_df <- merged_df %>%
        group_by_at(c(".draw", group_var, "time", "state")) %>%
        mutate(treated = sum(exp(mu_treated)), untreated = sum(exp(mu)), denom_val = mean(.data[[denom]] * years)) %>%
        ungroup()
      
      # Create Ban States aggregates INSIDE the function like dobbs_fertility  
      ban_states_df <- state_df %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = ifelse("D" %in% names(state_df), max(state_df$D) + 1, 999), state = "Ban States", banned_state = TRUE)
      
      ban_states_no_tx_df <- state_df %>%
        filter(state != "Texas" & state != "TX") %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = ifelse("D" %in% names(state_df), max(state_df$D) + 2, 1000), state = "Ban States (excl. Texas)", banned_state = TRUE)
      
      # Use the fresh aggregates if they don't exist, or the pre-computed ones if they do
      if (has_ban_aggregates) {
        # For pre-computed aggregates, handle differently based on whether state is an aggregate
        working_df <- merged_df_orig %>%
          filter(state %in% states) %>%
          group_by_at(c(".draw", group_var, "time", "state")) %>%
          mutate(
            # For aggregate states (Ban States, Ban States excl. Texas), mu/mu_treated are already log(sum(exp(...)))
            # So we just exponentiate them. For regular states, we sum across categories.
            is_aggregate = state %in% c("Ban States", "Ban States (excl. Texas)"),
            treated = ifelse(first(is_aggregate), sum(exp(mu_treated)), sum(exp(mu_treated))),
            untreated = ifelse(first(is_aggregate), sum(exp(mu)), sum(exp(mu))),
            denom_val = mean(.data[[denom]] * years)
          ) %>%
          ungroup()
      } else {
        state_df <- bind_rows(state_df %>% select_at(colnames(ban_states_df)), ban_states_df, ban_states_no_tx_df)
        working_df <- state_df %>% filter(state %in% states)
      }
      
      state_df <- working_df %>%
        group_by_at(c(".draw", group_var)) %>%
        summarize(
          treated_rate = sum(treated) / sum(denom_val) * rate_normalizer,
          untreated_rate = sum(untreated) / sum(denom_val) * rate_normalizer,
          causal_effect_diff = treated_rate - untreated_rate,
          causal_effect_ratio = 100 * (treated_rate / untreated_rate - 1),
          causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio),
          .groups = "drop"
        ) %>% ungroup()
      
    } else {
      # Deaths computation - similar pattern
      state_df <- merged_df %>%
        group_by_at(c(".draw", group_var, "time")) %>%
        mutate(treated = sum(exp(mu_treated)), untreated = sum(exp(mu)), denom_val = sum(.data[[denom]])) %>%
        ungroup()
      
      ban_states_df <- state_df %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = ifelse("D" %in% names(state_df), max(state_df$D) + 1, 999), state = "Ban States", banned_state = TRUE)
      
      ban_states_no_tx_df <- state_df %>%
        filter(state != "Texas" & state != "TX") %>%
        group_by_at(c(".draw", setdiff(group_var, "state"), "time")) %>%
        summarize(treated = sum(treated), untreated = sum(untreated), denom_val = sum(denom_val), .groups = "drop") %>%
        mutate(D = ifelse("D" %in% names(state_df), max(state_df$D) + 2, 1000), state = "Ban States (excl. Texas)", banned_state = TRUE)
      
      state_df <- bind_rows(state_df %>% select_at(colnames(ban_states_df)), ban_states_df, ban_states_no_tx_df)
      state_df <- state_df %>% filter(state %in% states)
      
      state_df <- state_df %>% group_by_at(c(".draw", group_var)) %>%
        summarize(
          treated_rate = sum(treated) / sum(denom_val) * rate_normalizer,
          untreated_rate = sum(untreated) / sum(denom_val) * rate_normalizer,
          causal_effect_diff = treated_rate - untreated_rate,
          causal_effect_ratio = treated_rate / untreated_rate,
          causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio),
          .groups = "drop"
        ) %>% ungroup()
    }
  }
  
  # Factor operations
  if ("state" %in% names(state_df)) {
    state_df <- state_df %>%
      mutate(state = factor(state)) %>%
      mutate(state = fct_reorder(state, causal_effect, .fun = median)) %>%
      mutate(state = fct_relevel(
        state,
        "Ban States (excl. Texas)",
        "Ban States"
      )) %>%
      mutate(state = fct_recode(state,
                                "States w/ bans" = "Ban States",
                                "States w/ bans (excl. Texas)" = "Ban States (excl. Texas)"
      ))
  }
  
  if ("category" %in% names(state_df)) {
    state_df <- state_df %>%
      mutate(category = fct_relevel(category, "Total", "US-born", "Foreign-born"))
  }
  
  state_df %>%
    ggplot(aes(x = !!sym(x_var), y = causal_effect, color = fct_rev(!!sym(color_group)))) +
    ggdist::stat_pointinterval(
      aes(alpha = after_stat(level)),
      position = "dodge", 
      .width = c(0.95, 0.67)
    ) +
    ggdist::scale_interval_alpha_continuous(range = c(0.75, 1)) +
    colorspace::scale_color_discrete_qualitative() +
    scale_alpha_manual(values = c(0.5, 1)) +
    theme_bw(base_size = 16) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ylab("Expected Percent Change") +
    xlab("") +
    guides(colour = guide_legend(reverse = TRUE)) + 
    coord_flip() + 
    theme(strip.text.y = element_blank())
}

#' Generate violin plots of differences between categories
#'
#' @param merged_df Data frame with merged draws and data
#' @param state State to analyze (default: "Texas")
#' @param treatment_date Optional treatment date
#' @param target Target variable (default: "births")
#' @param denom Denominator variable (default: "population")
#' @param rate_normalizer Rate normalizer (default: 1000)
#' @param estimand Estimand type: "diff" or "ratio" (default: "diff")
#' @return A ggplot object
make_violin_diffs <- function(merged_df, state = "Texas", treatment_date = NULL, 
                               target = "births", denom = "population",
                               rate_normalizer = 1000,
                               estimand = "diff") {
  
  merged_df <- merged_df %>%
    filter(state == !!state, exposure_code == 1) %>%
    group_by(state) %>%
    mutate(years = interval(min(start_date, na.rm = TRUE), max(end_date, na.rm = TRUE)) / years(1)) %>%
    ungroup()
  
  if (target == "births") {
    state_df <- merged_df %>%
      group_by_at(c(".draw", "category")) %>%
      mutate(outcome = sum(.data[[target]]), ypred = sum(ypred), denom_val = mean(.data[[denom]])) %>%
      mutate(
        outcome_rate = (outcome / years) / (denom_val / rate_normalizer),
        ypred_rate = ypred / years / (denom_val / rate_normalizer)
      ) %>%
      mutate(
        causal_effect_diff = outcome_rate - ypred_rate,
        causal_effect_ratio = outcome_rate / ypred_rate,
        causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio)
      ) %>%
      ungroup()
  } else {
    state_df <- merged_df %>%
      group_by_at(c(".draw", "category")) %>%
      mutate(outcome = sum(.data[[target]]), ypred = sum(ypred), denom_val = sum(.data[[denom]])) %>%
      mutate(outcome_rate = outcome / (denom_val / rate_normalizer),
             ypred_rate = ypred / (denom_val / rate_normalizer)) %>%
      mutate(causal_effect_ratio = outcome_rate / ypred_rate,
             causal_effect_diff = outcome_rate - ypred_rate,
             causal_effect = ifelse(estimand == "diff", causal_effect_diff, causal_effect_ratio)) %>%
      ungroup()
  }
  
  state_df <- state_df %>%
    mutate(category = factor(category)) %>%
    mutate(category = fct_reorder(category, causal_effect, .fun = median))
  
  cat_levels <- setdiff(levels(state_df$category), c("Total", "total"))
  diff_cats <- combn(cat_levels, 2) %>% t
  diff_cat1 <- diff_cats[, 2]
  diff_cat2 <- diff_cats[, 1]
  
  state_wide_df <- state_df %>%
    pivot_wider(
      id_cols = c(.draw, time), names_from = category,
      values_from = causal_effect
    )
  
  for (i in 1:length(diff_cat1)) {
    state_wide_df <- state_wide_df %>% 
      mutate(!!paste0(diff_cat1[i], " - ", diff_cat2[i]) := !!sym(diff_cat1[i]) - !!sym(diff_cat2[i]))
  }
  
  state_long_df <- state_wide_df %>% pivot_longer(
    cols = contains(" - "),
    names_to = "diff_cat", values_to = "diff"
  )
  
  state_long_df <- state_long_df %>%
    mutate(diff_cat = factor(diff_cat)) %>%
    mutate(diff_cat := fct_reorder(diff_cat, diff, .fun = median))   
  
  stats_df <- state_long_df %>%
    group_by(diff_cat) %>%
    summarize(
      mean = round(mean(diff), 3),
      pval = round(mean(diff < 0), 3),
      Significance = factor(ifelse(pval < 0.05, "red", "black"), levels = c("red", "black")),
      .groups = "drop"
    ) %>%
    ungroup() 
  
  state_long_df %>%
    ggplot(aes(x = diff_cat, y = diff)) +
    geom_violin(fill = "gray", alpha = 0.5, draw_quantiles = c(0.5)) +
    geom_hline(yintercept = 0, col = "red", linetype = "dashed") +
    geom_text(data = stats_df, aes(x = diff_cat, y = Inf, label = pval, col = Significance), 
              vjust = 2, fontface = "bold") +
    scale_colour_manual(values = c("red" = "red", "black" = "black")) + 
    guides(colour = "none") +
    geom_text(data = stats_df, aes(x = diff_cat, y = Inf, label = mean), vjust = 4, fontface = "bold") +
    theme_bw(base_size = 16) + 
    ggtitle("Difference in Group Effects") +
    xlab("State") + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) + 
    ylab("")
}

#' Posterior Predictive Check: ACF
#'
#' @param merged_df Data frame with merged draws and data
#' @param lag Lag for autocorrelation (default: 6)
#' @param outcome Outcome variable name (default: "births")
#' @param categories Optional vector of categories
#' @return List with pvals and acf_plt
make_acf_ppc_plot <- function(merged_df, lag = 6, 
                              outcome = "births",
                              categories = NULL) {
  
  if (is.null(categories)) {
    categories <- unique(merged_df$category)
  }
  
  ban_states <- merged_df %>%
    filter(banned_state == 1) %>%
    pull(state) %>%
    unique()
  
  acf_stats <- merged_df %>%
    filter(exposure_code == 0, state %in% ban_states) %>%
    filter(state != "Ban States") %>%
    filter(category %in% categories) %>%
    mutate(pred_diff = ypred - exp(mu)) %>%
    mutate(obs_diff = .data[[outcome]] - exp(mu)) %>%
    group_by(state, category, .draw) %>%
    summarise(
      obs_ac = acf(obs_diff, lag.max = lag, plot = FALSE)$acf[lag + 1, 1, 1],
      pred_ac = acf(pred_diff, lag.max = lag, plot = FALSE)$acf[lag + 1, 1, 1],
      diff_in_ac = obs_ac - pred_ac,
      .groups = "drop"
    )
  
  pvals <- acf_stats %>%
    group_by(state, category) %>%
    summarize(pval = mean(diff_in_ac < 0), .groups = "drop") %>%
    ungroup() %>%
    filter(category %in% categories) %>%
    filter(state %in% ban_states)
  
  acf_plt <- acf_stats %>%
    filter(category %in% categories) %>%
    ggplot() +
    geom_histogram(aes(x = diff_in_ac), alpha = 0.5) +
    geom_text(data = pvals, aes(label = round(pval, 3)), 
              y = Inf, x = Inf, hjust = 1, vjust = 1, col = "red") +
    geom_vline(xintercept = 0, col = "red", linetype = "dashed") + 
    ggtitle(sprintf("Difference in Residual Autocorrelation (Lag %i)", lag)) +
    facet_wrap(~state + category, scales = "free", ncol = 3) + 
    theme_bw() + 
    xlab("Observed - Predicted Autocorrelation")
  
  list("pvals" = pvals$pval, "acf_plt" = acf_plt)
}

#' Posterior Predictive Check: RMSE
#'
#' @param merged_df Data frame with merged draws and data
#' @param outcome Outcome variable name (default: "births")
#' @param categories Optional vector of categories
#' @return List with pvals and rmse_plt
make_rmse_ppc_plot <- function(merged_df, 
                               outcome = "births", categories = NULL) {
  
  if (is.null(categories)) {
    categories <- unique(merged_df$category)
  }
  
  rmse_stats <- merged_df %>%
    filter(exposure_code == 0) %>% 
    mutate(pred_diff = ypred - exp(mu)) %>%
    mutate(obs_diff = .data[[outcome]] - exp(mu)) %>%
    group_by(state, category, .draw) %>%
    summarise(
      rmse_pred_diff = sqrt(mean(pred_diff^2)),
      rmse_obs_diff = sqrt(mean(obs_diff^2)),
      .groups = "drop"
    ) %>% 
    mutate(diff_in_diff = rmse_obs_diff - rmse_pred_diff)
  
  ban_states <- merged_df %>%
    filter(banned_state == 1) %>%
    pull(state) %>%
    unique()
  
  pvals <- rmse_stats %>%
    group_by(state, category) %>%
    summarize(pval = mean(diff_in_diff < 0), .groups = "drop") %>%
    ungroup() %>%
    filter(category %in% categories) %>%
    filter(state %in% ban_states)
  
  rmse_plt <- rmse_stats %>% 
    filter(state %in% ban_states) %>%
    filter(category %in% categories) %>%
    ggplot() +
    geom_histogram(aes(x = diff_in_diff), alpha = 0.5) +
    geom_text(data = pvals, aes(label = round(pval, 3)), 
              y = Inf, x = Inf, hjust = 1, vjust = 1, col = "red") +
    geom_vline(xintercept = 0, col = "red", linetype = "dashed") +
    facet_wrap(~ state + category, scales = "free", ncol = 3) +
    theme_bw() +
    ggtitle("Difference in RMSE") + 
    xlab("Observed - Predicted RMSE")
  
  list("pvals" = pvals$pval, "rmse_plt" = rmse_plt)
}

#' Posterior Predictive Check: Maximum Absolute Residual
#'
#' @param merged_df Data frame with merged draws and data
#' @param outcome Outcome variable name (default: "births")
#' @param categories Optional vector of categories
#' @return List with pvals and max_plt
make_abs_res_ppc_plot <- function(merged_df,  
                                  outcome = "births", categories = NULL) {
  
  if (is.null(categories)) {
    categories <- unique(merged_df$category)
  }
  
  max_stats <- merged_df %>%
    filter(exposure_code == 0) %>% 
    mutate(pred_diff = ypred - exp(mu)) %>%
    mutate(obs_diff = .data[[outcome]] - exp(mu)) %>%
    group_by(state, category, .draw) %>%
    summarise(
      max_pred_diff = max(abs(pred_diff)),
      max_obs_diff = max(abs(obs_diff)),
      .groups = "drop"
    ) %>% 
    mutate(diff_in_diff = max_obs_diff - max_pred_diff)
  
  ban_states <- merged_df %>%
    filter(banned_state == 1) %>%
    pull(state) %>%
    unique()
  
  pvals <- max_stats %>%
    group_by(state, category) %>%
    summarize(pval = mean(diff_in_diff < 0), .groups = "drop") %>%
    ungroup() %>%
    filter(category %in% categories) %>%
    filter(state %in% ban_states)
  
  max_plt <- max_stats %>%
    filter(state %in% ban_states) %>%
    filter(category %in% categories) %>%
    ggplot() +
    geom_histogram(aes(x = diff_in_diff), alpha = 0.5) +
    geom_text(data = pvals, aes(label = round(pval, 3)), 
              y = Inf, x = Inf, hjust = 1, vjust = 1, col = "red") +
    geom_vline(xintercept = 0, col = "red", linetype = "dashed") +
    facet_wrap(~ state + category, scales = "free", ncol = 3) +
    theme_bw() +
    ggtitle("Difference in Maximum Absolute Predicted Residual") + 
    xlab("Observed - Predicted Max Residual")
  
  list("pvals" = pvals$pval, "max_plt" = max_plt)
}

#' Posterior Predictive Check: Unit Correlation
#'
#' @param merged_df Data frame with merged draws and data
#' @param max_treat_date Maximum treatment date (default: "2022-04-01")
#' @param categories Optional vector of categories
#' @param ndraws_to_use Number of draws to use (default: 1000)
#' @param outcome Outcome variable name (default: "births")
#' @return List with pvals and eval_plt
make_unit_corr_ppc_plot <- function(merged_df,
                                    max_treat_date = "2022-04-01", 
                                    categories = NULL,
                                    ndraws_to_use = 1000, 
                                    outcome = "births") {
  
  if (is.null(categories)) {
    categories <- unique(merged_df$category)
  }                           
  
  eval_stats <- merged_df %>%
    filter(time < max_treat_date) %>%
    filter(.draw < ndraws_to_use) %>%
    filter(category %in% categories) %>%
    mutate(obs_residual = .data[[outcome]] - exp(mu), pred_residual = ypred - exp(mu)) %>%
    group_by(state, category) %>%
    mutate(na_outcomes = mean(is.na(.data[[outcome]]))) %>%
    ungroup() %>%
    filter(na_outcomes < 0.25) %>%
    group_by(category, .draw) %>%
    summarise(
      obs_sval = sqrt(eigen(cor(matrix(obs_residual, ncol = length(unique(D))), 
                                use = "pairwise.complete.obs"))$values[1]),
      pred_sval = sqrt(eigen(cor(matrix(as.logical(obs_residual) * pred_residual, ncol = length(unique(D))), 
                                 use = "pairwise.complete.obs"))$values[1]),
      .groups = "drop"
    ) %>%
    mutate(eval_diff = obs_sval - pred_sval)
  
  pvals <- eval_stats %>%
    group_by(category) %>%
    summarize(pval = mean(eval_diff < 0), .groups = "drop") %>%
    ungroup() %>%
    filter(category %in% categories)
  
  eval_plt <- eval_stats %>%
    ggplot() +
    geom_histogram(aes(x = eval_diff), alpha = 0.5) +
    geom_text(data = pvals, aes(label = round(pval, 3)), 
              y = Inf, x = Inf, hjust = 1, vjust = 1, col = "red") +
    geom_vline(xintercept = 0, col = "red", linetype = "dashed") +
    facet_wrap(~category, scales = "free", ncol = 2) +
    theme_bw() +
    ggtitle("Difference in State Correlations") +
    xlab("Observed - Predicted Spectral Norm")
  
  list("pvals" = pvals$pval, "eval_plt" = eval_plt)
}

#' Generate fertility table with observed vs expected births
#'
#' @param merged_df Data frame with merged draws and data
#' @param target_state State to analyze (default: "Texas")
#' @param target Target variable (default: "births")
#' @param denom Denominator variable (default: "population")
#' @param rate_normalizer Rate normalizer (default: 1000)
#' @param plot_type Plot type (default: "exploratory")
#' @param tab_caption Table caption
#' @return A gt table object
make_fertility_table <- function(merged_df, 
                                 target_state = "Texas", 
                                 target = "births", 
                                 denom = "population",
                                 rate_normalizer = 1000, 
                                 plot_type = "exploratory",
                                 tab_caption = "Table 1. Estimated difference in cumulative observed vs expected births (count and rate) by nativity.") {
  
  if (target_state == "Ban States") {
    merged_df <- merged_df %>% filter(!state %in% c("Ban States", "Ban States (excl. Texas)"))
    merged_df <- merged_df %>%
      filter(exposure_code == 1) %>%
      group_by(type, category, .draw, time) %>% 
      summarise(
        !!target := sum(.data[[target]]), 
        denom_val = sum(.data[[denom]]), 
        ypred = sum(ypred), 
        mu = log(sum(exp(mu))),
        mu_treated = log(sum(exp(mu_treated))),
        years = mean(interval(start_date, end_date) / years(1)),
        .groups = "drop"
      )
  } else if (target_state == "Ban States (excl. Texas)") {
    merged_df <- merged_df %>% filter(!state %in% c("Ban States", "Ban States (excl. Texas)"))
    merged_df <- merged_df %>%
      filter(state != "Texas" & state != "TX") %>%
      filter(exposure_code == 1) %>%
      group_by(type, category, .draw, time) %>% 
      summarise(
        !!target := sum(.data[[target]]), 
        denom_val = sum(.data[[denom]]), 
        ypred = sum(ypred), 
        mu = log(sum(exp(mu))),
        mu_treated = log(sum(exp(mu_treated))),
        years = mean(interval(start_date, end_date) / years(1)),
        .groups = "drop"
      )
  } else {
    merged_df <- merged_df %>%
      filter(state == target_state, exposure_code == 1) %>%
      mutate(years = interval(start_date, end_date) / years(1), denom_val = .data[[denom]])
  }
  
  table_df <- merged_df %>%
    ungroup() %>%
    group_by(type, category, .draw) %>%
    summarize(
      ypred = sum(ypred),
      outcome = sum(.data[[target]]), 
      years = mean(years),
      treated = sum(exp(mu_treated)), 
      untreated = sum(exp(mu)),
      denom = sum(denom_val * years, na.rm = TRUE),
      treated_rate = treated / denom * rate_normalizer,
      untreated_rate = untreated / denom * rate_normalizer,
      outcome_rate = round(outcome / denom * rate_normalizer, 2),
      outcome_diff = round(treated - untreated),
      .groups = "drop"
    ) %>%
    ungroup() %>%
    group_by(type, category) %>%
    summarize(
      ypred_mean = mean(ypred),
      outcome = mean(outcome),
      outcome_diff_mean = round(mean(outcome_diff)), 
      outcome_diff_lower = round(quantile(outcome_diff, 0.025)), 
      outcome_diff_upper = round(quantile(outcome_diff, 0.975)),
      outcome_rate = mean(outcome_rate),
      treated_mean = mean(treated), 
      untreated_mean = mean(untreated),      
      treated_rate_mean = mean(treated_rate), 
      untreated_rate_mean = mean(untreated_rate), 
      causal_effect_diff_mean = mean(treated_rate - untreated_rate), 
      causal_effect_diff_lower = quantile(treated_rate - untreated_rate, 0.025), 
      causal_effect_diff_upper = quantile(treated_rate - untreated_rate, 0.975),
      causal_effect_ratio_mean = mean(treated_rate / untreated_rate), 
      causal_effect_ratio_lower = quantile(treated_rate / untreated_rate, 0.025), 
      causal_effect_ratio_upper = quantile(treated_rate / untreated_rate, 0.975),
      denom = mean(denom),
      pval = 2 * min(mean(untreated_rate > treated_rate), mean(untreated < treated)),
      .groups = "drop"
    )
  
  table_df <- table_df %>%
    mutate(
      rate_diff = round(causal_effect_diff_mean, 2),
      rate_diff_lower = round(causal_effect_diff_lower, 2),
      rate_diff_upper = round(causal_effect_diff_upper, 2),
      mult_change = causal_effect_ratio_mean,
      mult_change_lower = causal_effect_ratio_lower,
      mult_change_upper = causal_effect_ratio_upper
    )
  
  table_df <- table_df %>%
    mutate(birth_counts_str = paste0(outcome_diff_mean, " (", outcome_diff_lower, ", ", outcome_diff_upper, ")")) %>%
    mutate(birth_rate_abs_str = paste0(rate_diff, " (", rate_diff_lower, ", ", rate_diff_upper, ")")) %>%
    mutate(birth_rate_pct_str = paste0(round(100 * (mult_change - 1), 2), " (", round(100 * (mult_change_lower - 1), 2), ", ", round(100 * (mult_change_upper - 1), 2), ")")) %>%
    ungroup() %>%
    # Filter: for total type show only total category (lowercase), for any stratified type (nativity, groups) show only non-total
    filter((type == "total" & category == "total") | (type != "total" & category != "total")) %>%
    mutate(category = fct_relevel(category, "total", "usborn", "foreign", "hisp_usborn", "hisp_foreign", "nh_usborn", "nh_foreign"))
  
  pvals <- pval_rows <- table_df %>% pull(pval)
  pval_rows <- which(pvals < 0.05)
  table_df <- table_df %>% mutate(category = paste0(category, ifelse(pval <= 0.05, "*", "")))
  
  # Create row grouping by type (like dobbs creates row groups by demographic category)
  table_gt <- table_df %>%
    select(type, category, denom, outcome, outcome_rate, outcome_diff_mean, rate_diff, birth_counts_str, birth_rate_abs_str, birth_rate_pct_str) %>%
    mutate(expected_outcome = outcome - outcome_diff_mean, expected_rate = outcome_rate - rate_diff) %>%
    select(-c("outcome_diff_mean", "rate_diff")) %>%
    gt(rowname_col = "category") %>%
    tab_header(title = tab_caption)
  
  # Only add row grouping if we have groups type data
  if (any(table_df$type == "groups")) {
    table_gt <- table_gt %>%
      tab_row_group(label = "Subgroups", rows = type == "groups")
    
    # Only add row_group_order if we have both total and groups
    if (any(table_df$type == "total") && any(table_df$type == "groups")) {
      table_gt <- table_gt %>% row_group_order(groups = c(NA, "Subgroups"))
    }
  }
  
  table_gt %>%
    tab_spanner(
      label = "Fertility rate",
      columns = c(outcome_rate, expected_rate, birth_rate_abs_str, birth_rate_pct_str)
    ) %>%
    tab_spanner(
      label = "Birth count",
      columns = c(outcome, expected_outcome, birth_counts_str)
    ) %>%
    cols_label(
      denom = "Person-Years",
      outcome_rate = "Observed",
      expected_rate = "Expected",
      birth_rate_abs_str = html("Expected difference<br>(95% CI)"),
      birth_rate_pct_str = html("Expected percent change<br>(95% CI)"),
      outcome = "Observed",
      expected_outcome = "Expected",
      birth_counts_str = html("Expected difference<br>(95% CI)"),
      category = ""
    ) %>%
    tab_stub_indent(rows = category != "Total*" & category != "Total", indent = 5) %>%
    tab_options(table.align = "left", heading.align = "left") %>%
    cols_align(align = "left") %>%
    cols_hide(c(type, category)) %>%
    tab_options(table.font.size = 8) %>%
    opt_vertical_padding(scale = 0.5) %>%
    cols_width(
      category ~ px(125),
      birth_rate_abs_str ~ px(100),
      birth_rate_pct_str ~ px(100),
      outcome_rate ~ px(50),
      expected_rate ~ px(50),
      expected_outcome ~ px(50),
      birth_counts_str ~ px(100),
      outcome ~ px(50),
      denom ~ px(60)
    )
}
