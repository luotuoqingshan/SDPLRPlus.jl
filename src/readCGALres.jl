using MAT
using JSON
using Printf

problem = "MaxCut"
SketchyCGALres_folder = homedir()*"/SketchyCGAL/results/Dual"*problem*"/"
SDPLRres_folder = homedir()*"/SDPLR-jl/output/"*problem*"/"
CSDPres_folder = homedir()*"/SDPLR-jl/output/"*problem*"/"

@printf("Graph   Time of SketchyCGAL(s)    Time of SDPLR(s)     Time of CSDP(s)    Duality Bound of SketchyCGAL   Duality Bound of SDPLR\n")
for i = 41:67  
    SketchyCGALres = matread(SketchyCGALres_folder*"/G$i/SketchyCGAL-R-10-seed-0.mat")["out"]
    SDPLRres = JSON.parsefile(SDPLRres_folder*"/G$i/SDPLR-jl-seed-0.json")
    CSDPres = JSON.parsefile(CSDPres_folder*"/G$i/csdp-seed-0.json")
    @printf("G%02d   %20.2lf %20.2lf %20.2lf %30.2e %25.2e\n", i, SketchyCGALres["totalTime"],  SDPLRres["totaltime"], CSDPres["time"], SketchyCGALres["info"]["stopObj"][end], SDPLRres["rel_duality_bound"])
end