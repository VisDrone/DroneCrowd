This is an implementation of our CVPR 2011 paper:

Hamed Pirsiavash, Deva Ramanan, Charless C. Fowlkes, "Globally-Optimal Greedy Algorithms for Tracking a Variable Number of Objects," to appear in International Conference on Computer vision and Pattern Recognition (CVPR'11), Jun 2011.

We are using matlab/C implementation of part based object detector from http://people.cs.uchicago.edu/~pff/latent/
and also C implementation of push-relabel algorithm from http://www.igsystems.com/cs2/index.html (please note the copyright file in "3rd_party/cs2/")

To run, please call "main.m".
1. It will download ETHZ dataset (481MB) if not available (may take 30 minutes) 
2. It will run part-based object detector to detect humans if the result is no available in the cache (may take an hour using 8 CPU cores). You don't need to run it since the result is included in the cache.
3. It will build the graph
4. It will call three tracking algorithms(DP, DP with NMS in the loop, and push-relabel) and plot the scores.(Figure 7 on the paper)
5. It will draw detected bounding boxes for the result of "DP with NMS in the loop" algorithm and make a new video. (you may change it to show the results of DP or push-relabel algorithm)

Please note that since "successive shortest path (SSP)" and "push-relabel" algorithms are both optimum and have identical results, we do not include the implementation of SSP.


