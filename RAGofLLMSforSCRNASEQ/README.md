# Purpose
This model allows for standard scRNAseq analysis workflow with the addition of cell type annotation using the GPT LLM with Retrieval Augmented Generation (RAG) of PubMed articles for enhanced accuracy.

# File manifest :: description of each file

**'GPT.ipynb'** - Generates an annotation prediction using standard GPT4-o with no RAG

**'Get_PBMC_DAVID.ipynb'** - Pulls DAVID functional gene annotations from website to inform the LLM.

**'PBMC.ipynb'** - Standard scRNAseq pipeline using scanpy to generate clusters from scRNAseq data.

**scRAG.ipynb'** - Generates an annotation prediction using GPT4-o along with RAG implementation of PubMed article data

