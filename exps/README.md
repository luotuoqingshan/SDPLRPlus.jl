# Experiments

This README introduces the code that we use to perform the experiments in the paper.  

## Data
We mainly use [GSet](https://www.cise.ufl.edu/research/sparse/matrices/Gset/), [SNAP](https://snap.stanford.edu/data/index.html) and [DIMACS10](https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/index.html) datasets. You can manually download those you are particular interested in or use a script. We provide one example in `download.sh`. We also provide example data under `data/`. 

We perform some light data preprocessing, including flipping negative edges to cater the need to `SCAMS`, adding dummy nodes for Minimum Bisection problem when necessary, and removing self-loops for SNAP datasets. The functions we use are included in `data_preprocess.jl`. We also provide some tools to transform .mat file to SDPA/SDPLR file, see `data_utils.jl`, in case you want to use `SDPLR1.03`.

## Experiments related to SDPLRPlus
Currently we support 4 problems, MaxCut, MinimumBisection, LovaszTheta, CutNorm. You can test our solver on one graph using
```
cd SDPLRPlus.jl/exps; julia --project test.jl --graph G1 --problem MaxCut
```
You can pick the graph and the problem.

To more efficiently batch test the solver, you can use the batch testing tool we provide. Modify the first a few lines in `gen_batch_test.jl` and then run it. In this way you get a batch testing file called `batch_test.txt`. Now you can parallelly test via running  
```
cd SDPLRPlus.jl/exps; cat batch_test.txt | parallel --jobs 9 --timeout 28800 {}
```
Here `--timeout` specifies the time limit, and `--jobs` means the number of parallel jobs you want to run. Keep in mind that you should not set `--jobs` larger than the number of cores you have, which will downgrade the performance.

Note that in our code we explicitly enforce the solver to be **single-threaded** for fair benchmarking against other solvers and we did the same to other solvers. You can modify this.

## Other solvers
We maintain our codes for other solvers mainly in forked repos, and we try to make them self-contained by adding comments and READMEs. However, some may not contain detailed READMEs. If you have any question, feel free to contact me. Here is a list. 
- [SketchyCGAL](https://github.com/luotuoqingshan/SketchyCGAL)
- [SCAMS](https://github.com/luotuoqingshan/SCAMSv2)  
- [Max Cut using Manopt](https://github.com/luotuoqingshan/maxcut)
- CSDP: code is contained in `SDPLRPlus.jl/exps/exp_csdp`
- [RALM, Q_LSE, Q_LQH](https://github.com/luotuoqingshan/Optimization-on-manifolds-with-extra-constraints)
- SDPLR1.03: We download the c code from [SDPLR](https://sburer.github.io/projects.html) and slightly modified the code to make it compute relative primal infeasibility in Euclidean scaling like other solvers.   
- [USBS](https://github.com/luotuoqingshan/usbs)