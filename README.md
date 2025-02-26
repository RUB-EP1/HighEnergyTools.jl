# Tools of Statistical analysis in HEP

This julia package was created in a scope of data analysis course in Ruhr University Bochum (WS 2024/25).
It contains basis function for working with statistical distribution, sampling and fitting.

It's a not a package that does one thing, rather a collection of useful tools.
The package has many dependencies (see [Project.toml](Project.toml)), so in the future it might get tired apart into smaller packages.

## Installation

It's not a registered package. To install it, you can use the following command:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/mmikhasenko/HighEnergyTools.jl.git"))
#
using HighEnergyTools
```

## Related packages

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
- [AlgebraPDF.jl](https://github.com/mmikhasenko/AlgebraPDF.jl)
