import pandas as pd
import mygene

# === File path
input_file = "/ESL/ESL_MPRA/Figure_5/Effect_size/outputs/plots/unique_rbp_names.txt"
output_file = "/ESL/ESL_MPRA/Figure_5/Effect_size/outputs/plots/unique_rbp_names_with_aliases.csv"

# === Load gene names
genes = pd.read_csv(input_file, header=None)[0].astype(str).str.strip().tolist()

# === Initialize MyGene.info client
mg = mygene.MyGeneInfo()

# === Query MyGene.info for gene aliases
results = mg.querymany(
    genes,
    scopes="symbol,alias,ensemblgene",
    fields="symbol,name,alias,ensembl.gene",
    species="human"
)

# === Build dataframe
data = []
for rec in results:
    query = rec.get("query", "")
    symbol = rec.get("symbol", "")
    name = rec.get("name", "")
    aliases = rec.get("alias", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    alias_str = ";".join(sorted(set(aliases))) if aliases else ""
    data.append({
        "Query": query,
        "Official_Symbol": symbol,
        "Name": name,
        "Aliases": alias_str
    })

# === Save to CSV
df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"Saved gene aliases to: {output_file}")
