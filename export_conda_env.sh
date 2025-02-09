# Note - the crossplatform file should include channel "conda-forge", but the --from-history export 
# does not have it, so I add it manually.

conda env export --from-history > langchain_comments_crossplatform.yml; 
conda env export > langchain_comments.yml
