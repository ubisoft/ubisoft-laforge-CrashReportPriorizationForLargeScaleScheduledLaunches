© [2024] Ubisoft Entertainment. All Rights Reserved
# Crash Report Prioritization

Software crashes are high-impact defects that affect software reliability, causing applications to terminate unexpectedly. Crash prioritization focuses the attention of maintenance teams on incoming types of crashes that are likely to have a large impact. Prior approaches have been applied to software that is continuously released, whereas, in the video game context, releases are often rolled out in large-scale scheduled launches, where a studio will work for months or years on a new title before releasing it on a scheduled date. In that context, crash data from live players is not available until after release, which is often too late to react. 
In this study, we analyze post-release game crashes to identify temporal patterns that can inform strategies for prioritization them. 
We use data from a previously released title to detect such viral crashes in a new title by leveraging stack-trace similarity and Machine Learning (ML).

This repository includes our Online Appendix and our replication package. 
Replication package includes the following:


- 0. time-series-clustering.ipynb
- 1. get-codellama-embeddings.py
- 2. compute-similarity.py
- 3. train-ML-model.py
- 4. evaluate.ipynb
     
© [2024] Ubisoft Entertainment. All Rights Reserved
