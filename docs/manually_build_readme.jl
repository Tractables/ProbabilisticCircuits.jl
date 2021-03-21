using Literate

source_dir = "$(@__DIR__)/src"

"replace script includes with file content in Literate code"
function replace_includes(str)
    pat = r"include\(\"(.*)\"\)"
    m = match(pat, str)
    while !isnothing(m)
        str = replace(str, "$(m.match)" =>
                read("$source_dir/$(m[1])", String))
        m = match(pat, str)
    end
    str
end

"hide `#plot` lines in Literate code"
function hide_plots(str)
    str = replace(str, r"#plot (.)*[\n\r]" => "")
    replace(str, r"#!plot (.*)[\n\r]" => s"\g<1>\n")
end

"show `#plot` lines in Literate code"
function show_plots(str)
    str = replace(str, r"#!plot (.)*[\n\r]" => "")
    replace(str, r"#plot (.*)[\n\r]" => s"\g<1>\n")
end

Literate.markdown("$source_dir/README.jl", "$(@__DIR__)/../"; documenter=false, credit=false, execute=true, 
    preprocess = hide_plots ∘ replace_includes)