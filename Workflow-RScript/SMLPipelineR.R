#!/usr/bin/env Rscript
# =====================================================================
# Omics ML Pipeline (General, Tabular Data)
# Author: Your Name
# Version: 0.1.0
# Dependencies: caret, randomForest, xgboost, nnet, tidyverse, pROC, VennDiagram
# =====================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(randomForest)
  library(xgboost)
  library(nnet)
  library(pROC)
  library(VennDiagram)
})

# ----------- Utility --------------------------------------------------

metric_from_cm <- function(cm) {
  # caret::confusionMatrix stores metrics in $byClass and $overall
  byc <- as.list(cm$byClass)
  ov  <- as.list(cm$overall)
  tibble::tibble(
    Accuracy = as.numeric(ov[["Accuracy"]]),
    Kappa = as.numeric(ov[["Kappa"]]),
    Sensitivity = as.numeric(byc[["Sensitivity"]]),
    Specificity = as.numeric(byc[["Specificity"]]),
    `Balanced Accuracy` = as.numeric(byc[["Balanced Accuracy"]]),
    `Pos Pred Value` = as.numeric(byc[["Pos Pred Value"]]),
    `Neg Pred Value` = as.numeric(byc[["Neg Pred Value"]]),
    `Mcnemar P` = suppressWarnings(as.numeric(ov[["Mcnemar's Test P-Value"]]))
  )
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# ----------- Pipeline Functions --------------------------------------

load_data <- function(path, label_col, id_col = NULL) {
  message("Loading data: ", path)
  ext <- tools::file_ext(path)
  df <- switch(tolower(ext),
               "csv" = readr::read_csv(path, show_col_types = FALSE),
               "tsv" = readr::read_tsv(path, show_col_types = FALSE),
               "txt" = readr::read_delim(path, delim = "\t", show_col_types = FALSE),
               stop("Unsupported file extension: ", ext))
  if (!label_col %in% names(df)) stop("label_col not found in data.")
  if (!is.null(id_col) && !id_col %in% names(df)) stop("id_col not found in data.")
  df <- df %>% mutate(!!label_col := as.factor(.data[[label_col]])) %>% droplevels()
  df
}

preprocess_data <- function(df, label_col, mode = c("complete_case", "median_impute"),
                            log1p = FALSE, center_scale = TRUE) {
  mode <- match.arg(mode)
  y <- df[[label_col]]
  x <- df %>% select(-all_of(label_col))
  # Keep only numeric predictors for modeling
  numeric_mask <- purrr::map_lgl(x, is.numeric)
  x_num <- x[, numeric_mask, drop = FALSE]
  # Optional log1p
  if (log1p) x_num <- mutate_all(x_num, ~log1p(.x))
  # Missing handling
  if (mode == "complete_case") {
    keep <- stats::complete.cases(x_num) & !is.na(y)
    x_num <- x_num[keep, , drop = FALSE]
    y <- y[keep]
  } else {
    # Median impute via caret preProcess on predictors only
    pp <- caret::preProcess(x_num, method = "medianImpute")
    x_num <- predict(pp, x_num)
  }
  # Center/scale if requested
  if (center_scale) {
    pp2 <- caret::preProcess(x_num, method = c("center", "scale"))
    x_num <- predict(pp2, x_num)
  }
  out <- bind_cols(x_num, tibble::tibble(!!label_col := y)) %>% drop_na(all_of(label_col))
  list(data = out, predictors = colnames(x_num))
}

make_splits <- function(df, label_col, ratios = c(0.6, 0.7, 0.8, 0.9), seed = 123) {
  set.seed(seed)
  splits <- list()
  for (r in ratios) {
    idx <- caret::createDataPartition(df[[label_col]], p = r, list = FALSE)
    train_idx <- as.vector(idx)
    test_idx  <- setdiff(seq_len(nrow(df)), train_idx)
    splits[[paste0(round(r*100), "_split")]] <- list(train = train_idx, test = test_idx)
  }
  splits
}

feature_screen <- function(df, label_col, alpha = 0.05, max_keep = NA) {
  # Two-group t-tests for numeric predictors
  y <- df[[label_col]]
  x <- df %>% select(-all_of(label_col))
  numeric_mask <- purrr::map_lgl(x, is.numeric)
  x <- x[, numeric_mask, drop = FALSE]
  res <- purrr::map_df(colnames(x), function(feat) {
    a <- x[[feat]][y == levels(y)[1]]
    b <- x[[feat]][y == levels(y)[2]]
    tt <- try(stats::t.test(a, b), silent = TRUE)
    if (inherits(tt, "try-error")) return(tibble::tibble(feature = feat, p = NA_real_, stat = NA_real_))
    tibble::tibble(feature = feat, p = tt$p.value, stat = unname(tt$statistic))
  }) %>% arrange(p)
  res$padj <- p.adjust(res$p, method = "BH")
  keep <- res %>% filter(padj < alpha)
  if (!is.na(max_keep)) keep <- keep %>% slice_head(n = max_keep)
  list(screen_table = res, keep_features = keep$feature)
}

train_one <- function(train_df, label_col, method = c("rf","xgbTree","nnet"), tuneLength = 10, seed = 123) {
  method <- match.arg(method)
  set.seed(seed)
  ctrl <- caret::trainControl(method = "repeatedcv", number = 5, repeats = 2,
                              classProbs = TRUE, summaryFunction = twoClassSummary,
                              savePredictions = "final")
  # Ensure positive class is the first level (caret uses first level as "event" for ROC)
  y <- train_df[[label_col]]
  if (length(levels(y)) != 2) stop("Outcome must be binary factor.")
  # Relevel so that the first level is the 'positive' class (customizable here)
  # By default, make the first level the one with lower frequency to emphasize recall;
  # adjust as needed for your use-case.
  levs <- levels(y)
  counts <- table(y)
  pos <- names(sort(counts))[1]
  train_df[[label_col]] <- relevel(train_df[[label_col]], ref = pos)

  fit <- caret::train(
    reformulate(termlabels = setdiff(names(train_df), label_col), response = label_col),
    data = train_df,
    method = method,
    metric = "ROC",
    trControl = ctrl,
    tuneLength = tuneLength
  )
  fit
}

evaluate_one <- function(fit, test_df, label_col) {
  # Predictions
  probs <- predict(fit, newdata = test_df, type = "prob")
  pred  <- predict(fit, newdata = test_df, type = "raw")
  # Ensure same positive level as training
  positive_class <- fit$levels[1]
  cm <- caret::confusionMatrix(pred, test_df[[label_col]], positive = positive_class)
  metrics <- metric_from_cm(cm) %>% mutate(Model = fit$method, Positive = positive_class)
  list(confusion = cm, metrics = metrics, probs = probs, pred = pred)
}

var_importance <- function(fit, top_n = 20) {
  imp <- try(caret::varImp(fit, scale = TRUE), silent = TRUE)
  if (inherits(imp, "try-error")) return(tibble::tibble(Feature = character(), Importance = numeric()))
  vi <- imp$importance %>% tibble::rownames_to_column("Feature") %>% arrange(desc(Overall))
  if (!is.null(top_n)) vi <- vi %>% slice_head(n = top_n)
  vi
}

consensus_signature <- function(imp_list, top_n = 10) {
  top_sets <- lapply(imp_list, function(df) head(df$Feature, top_n))
  names(top_sets) <- names(imp_list)
  inter_all <- Reduce(intersect, top_sets)
  # Also return pairwise intersections
  list(top_sets = top_sets, intersect_all = inter_all)
}

# ----------- Orchestrator --------------------------------------------

run_pipeline <- function(data_path,
                         label_col,
                         id_col = NULL,
                         out_dir = "outputs",
                         screen_alpha = 0.05,
                         screen_top = NA,
                         preprocess_mode = c("complete_case", "median_impute"),
                         log1p = FALSE,
                         center_scale = TRUE,
                         split_ratios = c(0.6, 0.7, 0.8, 0.9),
                         models = c("rf","xgbTree","nnet"),
                         tuneLength = 10,
                         seed = 123) {

  ensure_dir(out_dir)
  df0 <- load_data(data_path, label_col = label_col, id_col = id_col)

  # Preprocess
  pp <- preprocess_data(df0, label_col = label_col, mode = preprocess_mode,
                        log1p = log1p, center_scale = center_scale)
  df <- pp$data
  predictors <- pp$predictors

  # Feature screening
  fs <- feature_screen(df, label_col = label_col, alpha = screen_alpha, max_keep = screen_top)
  keep_feats <- if (length(fs$keep_features) > 0) fs$keep_features else predictors
  readr::write_csv(fs$screen_table, file.path(out_dir, "feature_screening.csv"))

  # Use screened features
  df_use <- df %>% select(all_of(c(keep_feats, label_col)))

  # Splits
  splits <- make_splits(df_use, label_col = label_col, ratios = split_ratios, seed = seed)

  # Storage
  all_metrics <- list()
  all_importance <- list()

  # Loop over splits and models
  for (sp in names(splits)) {
    tr_idx <- splits[[sp]]$train
    te_idx <- splits[[sp]]$test
    train_df <- df_use[tr_idx, , drop = FALSE]
    test_df  <- df_use[te_idx, , drop = FALSE]

    for (m in models) {
      fit <- train_one(train_df, label_col = label_col, method = m, tuneLength = tuneLength, seed = seed)
      ev  <- evaluate_one(fit, test_df, label_col = label_col)
      vi  <- var_importance(fit, top_n = 50)

      # Save artifacts
      model_tag <- paste(sp, m, sep = "_")
      saveRDS(fit, file = file.path(out_dir, paste0("model_", model_tag, ".rds")))
      readr::write_csv(ev$metrics %>% mutate(Split = sp), file.path(out_dir, paste0("metrics_", model_tag, ".csv")))
      readr::write_csv(vi, file.path(out_dir, paste0("varimp_", model_tag, ".csv")))

      all_metrics[[model_tag]] <- ev$metrics %>% mutate(Split = sp, Model = m)
      all_importance[[model_tag]] <- vi
    }
  }

  # Aggregate metrics
  metrics_df <- dplyr::bind_rows(all_metrics)
  readr::write_csv(metrics_df, file.path(out_dir, "metrics_all_models.csv"))

  # Consensus signature (per model family across best split or all splits)
  # Here we compute per family using all splits:
  families <- unique(metrics_df$Model)
  consensus <- list()
  for (fam in families) {
    fam_ims <- all_importance[grepl(paste0("_", fam, "$"), names(all_importance))]
    consensus[[fam]] <- consensus_signature(fam_ims, top_n = 10)
    # Save simple text summary
    sink(file.path(out_dir, paste0("consensus_", fam, ".txt")))
    cat("Model family:", fam, "\n")
    cat("Top-10 sets per split:\n")
    print(consensus[[fam]]$top_sets)
    cat("\nIntersection across splits:\n")
    print(consensus[[fam]]$intersect_all)
    sink()
  }

  # Optional: simple Venn diagram for first three sets of a family (if available)
  for (fam in names(consensus)) {
    ts <- consensus[[fam]]$top_sets
    if (length(ts) >= 3) {
      first_three <- ts[1:3]
      venn.plot <- VennDiagram::venn.diagram(
        x = first_three,
        filename = NULL,
        fill = c("#FEE08B", "#D53E4F", "#3288BD"),
        alpha = 0.5,
        cat.cex = 0.8,
        cex = 0.8,
        main = paste("Top-10 Feature Overlap -", fam)
      )
      grDevices::png(filename = file.path(out_dir, paste0("venn_", fam, ".png")), width = 1400, height = 1000, res = 150)
      grid::grid.draw(venn.plot)
      grDevices::dev.off()
    }
  }

  message("Pipeline complete. Outputs written to: ", out_dir)
  invisible(list(metrics = metrics_df, consensus = consensus))
}

# ----------- CLI Entry Point -----------------------------------------

if (sys.nframe() == 0) {
  # Example CLI usage:
  # Rscript pipeline.R --data data/omics.csv --label outcome --out outputs
  args <- commandArgs(trailingOnly = TRUE)
  arg_list <- list()
  if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
      key <- gsub("^--", "", args[i])
      val <- args[i + 1]
      arg_list[[key]] <- val
    }
  }
  data_path <- arg_list[["data"]]
  label_col <- arg_list[["label"]]
  out_dir   <- arg_list[["out"]]
  if (is.null(data_path) || is.null(label_col)) {
    stop("Usage: Rscript pipeline.R --data <path.csv> --label <label_col> [--out outputs]")
  }
  if (is.null(out_dir)) out_dir <- "outputs"

  # Run with defaults
  run_pipeline(
    data_path   = data_path,
    label_col   = label_col,
    out_dir     = out_dir,
    screen_alpha = 0.05,
    preprocess_mode = "complete_case",
    log1p = FALSE,
    center_scale = TRUE,
    split_ratios = c(0.6, 0.7, 0.8, 0.9),
    models = c("rf","xgbTree","nnet"),
    tuneLength = 10,
    seed = 123
  )
}
