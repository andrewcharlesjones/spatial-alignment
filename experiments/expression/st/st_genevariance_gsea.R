library(magrittr)
library(dplyr)
source("~/Documents/beehive/rrr/PRRR/prrr/util/ensembl_to_gene_symbol.R")
source("~/Documents/beehive/rrr/PRRR/prrr/util/gsea_shuffle.R")

results_path <- "/Users/andrewjones/Documents/beehive/spatial_stitching/spatial-alignment/experiments/expression/st/out/st_avg_gene_variances.csv"
gene_vars <- read.csv(results_path)

GO_BP_FILE <- "~/Documents/beehive/gtex_data_sample/gene_set_collections/GO_biological_process.gmt"
HALLMARK_FILE <- "~/Documents/beehive/gtex_data_sample/gene_set_collections/h.all.v7.1.symbols.gmt"

gsc_bp <- piano::loadGSC(GO_BP_FILE)
gsc_hallmark <- piano::loadGSC(HALLMARK_FILE)

gene_scores <- gene_vars[,2]
gene_scores %<>% scale()
names(gene_scores) <- gene_vars[,1]

gsea_results <- run_permutation_gsea(gsc_file=HALLMARK_FILE, gene_stats=gene_scores, nperm = 1000, gsc = gsc_hallmark)
gsea_results %>% filter(padj <= 0.01) %>% arrange(padj)
write.csv(x = gsea_results[,c("pathway", "padj", "ES", "NES")] %>% as.data.frame(), file = "/Users/andrewjones/Documents/beehive/spatial_stitching/spatial-alignment/experiments/expression/st/out/st_avg_gene_variance_gsea_results.csv")

gsea_results <- run_permutation_gsea(gsc_file=GO_BP_FILE, gene_stats=gene_scores, nperm = 1000, gsc = gsc_bp)
gsea_results %>% filter(padj <= 0.01) %>% arrange(padj)

n_genes <- 20
hit_genes <- names(gene_scores)[order(-gene_scores)][1:n_genes]
gsea_out <- run_fisher_exact_gsea(gsc_file = geneset_file, gsc = gsc_hallmark, hit_genes = hit_genes %>% unique(), all_genes = names(gene_scores) %>% unique())
out <- gsea_out %>% arrange(adj_pval) %>% head(1)

split_pathway_name <- function(x){
  split <- strsplit(x, "_")[[1]] %>% as.character() %>% as.list()
  return(paste(split[2:length(split)], sep = " ", collapse = " "))
}
pathway_names <- lapply(FUN = split_pathway_name, X = out$pathway) %>% as.character()
out["pathway"] <- pathway_names
out["adj_pval"] <- round(out$adj_pval, digits=3)
out <- out[,c("pathway", "adj_pval")]
colnames(out) <- c("Pathway", "Adj. p-val")

tab <- formattable::formattable(out)
# write.csv(x = gsea_results[,c("pathway", "padj", "ES", "NES")] %>% as.data.frame(), file = "/Users/andrewjones/Documents/beehive/spatial_stitching/spatial-alignment/experiments/expression/st/out/st_avg_gene_variance_gsea_results.csv")


gsea_out <- run_fisher_exact_gsea(gsc_file = geneset_file, gsc = gsc_bp, hit_genes = hit_genes %>% unique(), all_genes = names(gene_scores) %>% unique())
gsea_out %>% arrange(adj_pval) %>% head(10)


for (ii in seq(1, n_components)) {
  
  # Extract component
  curr_component <- scale(V[,ii] %>% as.double(), center = T, scale = T)
  names(curr_component) <- gene_names
  
  # Run GSEA on this component
  gsea_results <- run_permutation_gsea(gsc_file=HALLMARK_FILE, gene_stats=curr_component, nperm = 1000, gsc = gsc_hallmark)
  
  write.csv(x = gsea_results[,c("pathway", "pval", "padj", "ES", "NES")], file = file.path(save_dir, sprintf("gsea_results_component_%s.csv", toString(ii))))
  
  # print(gsea_results[,c("pathway", "pval", "padj", "NES")] %>% arrange(padj) %>% head(5))
  
  ## Run Fisher's exact test
  n_genes <- 30
  hit_genes <- names(curr_component)[order(curr_component)][1:n_genes]
  gsea_out <- run_fisher_exact_gsea(gsc_file = geneset_file, gsc = gsc_hallmark, hit_genes = hit_genes %>% unique(), all_genes = names(curr_component) %>% unique())
  gsea_out <- gsea_out[gsea_out$adj_pval <= 0.05]
  if (ncol(gsea_out) > 0) {
    print(gsea_out[,c("pathway", "adj_pval")])
  }
  
  
  hit_genes <- names(curr_component)[order(-curr_component)][1:n_genes]
  gsea_out <- run_fisher_exact_gsea(gsc_file = geneset_file, gsc = gsc_hallmark, hit_genes = hit_genes %>% unique(), all_genes = names(curr_component) %>% unique())
  gsea_out <- gsea_out %>% filter(adj_pval <= 0.05)
  if (ncol(gsea_out) > 0) {
    print(gsea_out[,c("pathway", "adj_pval")])
  }
}

