import pandas as pd
import urllib.request
import gzip
import io

print("Downloading IRMA dataset from GEO (GSE14264)...")
url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14264/matrix/GSE14264_series_matrix.txt.gz"

try:
    response = urllib.request.urlopen(url)
    compressed_file = io.BytesIO(response.read())
    decompressed_file = gzip.GzipFile(fileobj=compressed_file)
    
    # Parse the series matrix
    lines = decompressed_file.read().decode('utf-8').splitlines()
    data_lines = [line for line in lines if not line.startswith('!') and line.strip()]
    header = [line for line in lines if line.startswith('!series_matrix_table_begin')]
    # We want the header and the data
    
    with open("irma_expression_matrix_full.txt", "w") as f:
        for line in lines:
            if line.startswith('!Sample_title') or not line.startswith('!'):
                f.write(line + "\n")
                
    print("Successfully downloaded irma_expression_matrix_full.txt!")
    print("Note: This matrix contains all yeast microarray probes.")
    print("You will need to map the probes to the 5 IRMA genes: CBF1, GAL4, SWI5, GAL80, ASH1.")
    print("Alternatively, download the clean 'Table S1' directly from the Cell 2009 Cantone et al. supplementary materials.")
except Exception as e:
    print(f"Failed to download: {e}")
    print("Please download the supplementary dataset directly from the Cell publication (Cantone et al. 2009).")
