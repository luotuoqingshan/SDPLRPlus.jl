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
    kwargs...,
  )
    axis.xlabel = xlabel 
    axis.ylabel = ylabel
    xlims!(axis, (logscale ? 0.0 : 1.0, 1.1 * max_ratio))
    ylims!(axis, (0, 1.1))
    plots = []
    for i = 1:length(labels)
        push!(plots, CairoMakie.stairs!(axis, x_plot[i], y_plot[i],
         step=:post, label = labels[i], color=colors[i];))  # add to initial plot
    end
    if logscale
        if inv
            axis.xtickformat = values -> ["$(@sprintf("%.2f", 1/2^value))" for value in values] 
        else
            axis.xtickformat = values -> [L"2^{%$(Int64(value))}" for value in values]
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
    labels::Vector{S} = String[];
    logscale::Bool = true,
    title::AbstractString = "",
    sampletol::Float64 = 0.0,
    drawtol::Float64 = 0.0,
    inv::Bool = false,
    kwargs...,
) where {S <: AbstractString}
  (x_plot, y_plot, max_ratio) =
    performance_profile_data(T, logscale = logscale, sampletol = sampletol, drawtol = drawtol)
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