using BenchmarkProfiles, CairoMakie, Printf

function Makie_performance_profile_plot!(
    axis,
    x_plot,
    y_plot,
    max_ratio,
    inv,
    colors,
    xlabel,
    ylabel,
    labels,
    title,
    logscale;
    custom_xtickformat=nothing,
    kwargs...,
)
    axis.xlabel = xlabel
    axis.ylabel = ylabel
    xlims!(axis, (logscale ? 0.0 : 1.0, 1.1 * max_ratio))
    ylims!(axis, (0, 1.1))
    plots = []
    for i in 1:length(labels)
        push!(
            plots,
            CairoMakie.stairs!(
                axis,
                x_plot[i],
                y_plot[i];
                step=:post,
                label=labels[i],
                color=colors[i],
            ),
        )  # add to initial plot
    end
    if custom_xtickformat !== nothing
        axis.xtickformat = custom_xtickformat
        return plots
    end
    # currently only support logscale for x-axis
    if logscale
        if inv
            # inv = true means for this metric, larger is better 
            axis.xtickformat =
                values -> ["$(@sprintf("%.2f", 1/2^value))" for value in values]
        else
            axis.xtickformat =
                values -> [L"2^{%$(Int64(value))}" for value in values]
        end
    end
    return plots
end

function Makie_performance_profile(
    axis::Axis,
    T::Matrix{Float64},
    colors,
    xlabel,
    ylabel,
    labels::Vector{S}=String[];
    logscale::Bool=true,
    title::AbstractString="",
    sampletol::Float64=0.0,
    drawtol::Float64=0.0,
    inv::Bool=false,
    kwargs...,
) where {S<:AbstractString}
    (x_plot, y_plot, max_ratio) = performance_profile_data(
        T; logscale=logscale, sampletol=sampletol, drawtol=drawtol
    )
    return Makie_performance_profile_plot!(
        axis,
        x_plot,
        y_plot,
        max_ratio,
        inv,
        colors,
        xlabel,
        ylabel,
        labels,
        title,
        logscale;
        kwargs...,
    )
end

### Example
#using Colors
#
#plot_height = 300
#mylabelsize = 9
#n = 3
#
#cols = distinguishable_colors(n, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
#solvers = ["solver$i" for i=1:n]
#dts = 10 * abs.(rand(25, n))
#f=Figure(; size=(plot_height, plot_height), figure_padding=0) # set padding
#axis = Axis(f[1, 1], 
#    alignmode=Outside(),
#    xticklabelsize=mylabelsize,yticklabelsize=mylabelsize,
#    xticksize=3,yticksize=3,
#    xlabelsize=mylabelsize, ylabelsize=mylabelsize)
#hidespines!(axis)
#rowgap!(f.layout, 0)  
#
#Makie_performance_profile(axis, dts, cols, 
#    L"\tau", 
#    "Proportions",
#    solvers,
#    logscale=true)
#xlims!(axis, (0, 11))
#f
