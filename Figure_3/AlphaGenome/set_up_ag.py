from alphagenome.data import genome
from alphagenome.models import dna_client
import numpy as np

API_KEY = "your API key"
model = dna_client.create(API_KEY)

seq = 'AGGTAAGTTAACCAG'.center(dna_client.SEQUENCE_LENGTH_1MB, 'N')

output = model.predict_sequence(
    sequence=seq,
    requested_outputs=[dna_client.OutputType.SPLICE_SITE_USAGE],
    ontology_terms=['UBERON:0001157'],
)
print("Splice site usage shape:", output.splice_site_usage.values.shape)
print("Sample values (center ±5):", output.splice_site_usage.values[524283:524293])

output = model.predict_sequence(
    sequence=seq,
    requested_outputs=[dna_client.OutputType.SPLICE_SITE_USAGE],
    ontology_terms=['UBERON:0000955'],
)
print(output.splice_site_usage.metadata.columns.tolist())
print(output.splice_site_usage.metadata.head(10).to_string())
print(output.splice_site_usage.metadata[['ontology_curie', 'biosample_name', 'biosample_type']].drop_duplicates().to_string())

# Get ALL available ontology terms without making a prediction
output_metadata = model.output_metadata(
    organism=dna_client.Organism.HOMO_SAPIENS
)
splice_meta = output_metadata.splice_site_usage
print("\n--- All available SPLICE_SITE_USAGE ontologies ---")
print(splice_meta[['ontology_curie', 'biosample_name', 'biosample_type']].drop_duplicates().to_string())


out1 = model.predict_sequence(
    sequence=seq,
    requested_outputs=[dna_client.OutputType.SPLICE_SITES],
    ontology_terms=['UBERON:0000955'],  # brain
)

out2 = model.predict_sequence(
    sequence=seq,
    requested_outputs=[dna_client.OutputType.SPLICE_SITES],
    ontology_terms=['UBERON:0002048'],  # lung
)

import numpy as np

ontologies = [
    ('UBERON:0000955', 'brain'),
    ('UBERON:0002048', 'lung'),
    ('CL:0000127', 'astrocyte'),
    ('EFO:0002067', 'K562'),
    ('UBERON:0002107', 'liver'),
]

results = {}
for term, name in ontologies:
    out = model.predict_sequence(
        sequence=seq,
        requested_outputs=[dna_client.OutputType.SPLICE_SITES],
        ontology_terms=[term],
    )
    results[name] = out.splice_sites.values
    print(f"{name}: shape {out.splice_sites.values.shape}")

# Compare all against brain
ref = results['brain']
for name, vals in results.items():
    if name == 'brain':
        continue
    diff = np.abs(ref - vals)
    print(f"brain vs {name} -- max diff: {diff.max():.2e}, identical: {np.array_equal(ref, vals)}")