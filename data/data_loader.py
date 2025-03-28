import scanpy as sc
import os

# Load the h5ad file
adata = sc.read_h5ad(r"D:\AI_DS_research\Github_Projects\Inti-Internship\data\raw\GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad")

# View the dataset structure
print(adata)

# Convert to Pandas DataFrame if needed
gene_expression_df = adata.to_df()
print(gene_expression_df.head())

# Convert to DataFrame
gene_expression_df = adata.to_df()

# Define path to save the CSV file inside the processed folder
output_path = os.path.join(os.path.dirname(__file__), "raw", "snRNA_seq_data.csv")

# Save as CSV
# Save as CSV without the unnamed first column
gene_expression_df.to_csv(output_path, index=False)
print(f"CSV file saved at: {output_path}")