using Pkg # for 0.7+
using Logging
SetLevel
Pkg.add("PyCall")
Pkg.add(PackageSpec("https://github.com/malmaud/TensorFlow.jl.git") )
# Pkg.build("TensorFlow")
Pkg.add("IJulia")
using TensorFlow
using IJulia
