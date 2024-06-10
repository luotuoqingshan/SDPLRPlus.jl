# SDPLRPlus

<!-- Tidyverse lifecycle badges, see https://www.tidyverse.org/lifecycle/ Uncomment or delete as needed. -->
![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![build](https://github.com/luotuoqingshan/SDPLR.jl/workflows/CI/badge.svg)](https://github.com/luotuoqingshan/SDPLR.jl/actions?query=workflow%3ACI)
<!-- travis-ci.com badge, uncomment or delete as needed, depending on whether you are using that service. -->
<!-- [![Build Status](https://travis-ci.com/luotuoqingshan/SDPLR.jl.svg?branch=master)](https://travis-ci.com/luotuoqingshan/SDPLR.jl) -->
<!-- Coverage badge on codecov.io, which is used by default. -->
[![codecov.io](http://codecov.io/github/luotuoqingshan/SDPLR.jl/coverage.svg?branch=master)](http://codecov.io/github/luotuoqingshan/SDPLR.jl?branch=master)
<!-- Documentation -- uncomment or delete as needed -->
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://luotuoqingshan.github.io/SDPLR.jl/stable)
-->
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://luotuoqingshan.github.io/SDPLR.jl/dev)
<!-- Aqua badge, see test/runtests.jl -->
<!-- [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) -->

SDPLRPlus is a pure julia package which integrates the suboptimality bound and dynamic rank update idea introduced in the following paper with the original [SDPLR](https://sburer.github.io/projects.html) solver.  
```
Suboptimality bounds for trace-bounded SDPs enable
a faster and scalable low-rank SDP solver SDPLR+
```
If you feel this package or our paper useful, please consider cite it. 

For information with regard to the package, refer to the documentation for more details.  

## Experiments 
For more information about experiments in the paper, 
take a look at the README under `exps/`. 

## Contact
Feel free to contact via email huan1754 at purdue dot edu or create issues or pull requests (any contribution to the project is welcome).

## Acknowledgement
Part of this project started as a julia reproduction of 
the solver [SDPLR](https://sburer.github.io/projects.html) and we would like to thank the authors 
for their effort in developing such an amazing package.  
Please consider citing their papers if you use 
this package in your research, see reference section
in the documentation for more information.