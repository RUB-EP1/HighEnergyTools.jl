(delete this file before merging to main)

# SPlot Refactoring - Key Conceptual Questions

## 1. **SPlot Structure Scope**
Your example shows `SPlot(model)` with a `MixtureModel`. Should this:
- Work with any `MixtureModel` (arbitrary number of components)?
- Or be specialized for 2-component signal/background cases?

The current code is hardcoded for 2 components - should the refactored version be more general?

## 2. **Weight Matrix Caching Strategy**
The weight matrix `W` and its inverse are expensive to compute. Should `SPlot`:
- Compute and cache `W` and `α = inv(W)` once during construction?
- Or compute them on-demand when `sweights()` is called?

## 3. **Component Access Pattern**
Your example shows `view(w, 1, :)` suggesting a matrix return. Should `sweights()` return:
- A matrix `w[component, event]` where rows are components?
- Or a more structured object with component names?

## 4. **Fitting Integration**
Should `SPlot`:
- Take a pre-fitted model with known yields (like your example)?
- Or automatically fit the model to data during construction?
- Or have a separate `fit!(splot, data)` method?

## 5. **MixtureModel Weight Handling**
When you pass a `MixtureModel([signal, background], [0.6, 0.4])`, should `SPlot`:
- Use the component weights `[0.6, 0.4]` as the fitted yields?
- Or treat these as initial guesses and fit the actual yields to data?

## 6. **Variance Calculation Scope**
Should variance calculation be:
- Built into the `SPlot` object and always available?
- Or a separate method that requires additional computation?

---

## Proposed API Example
```julia
model_range = (5.0, 5.6)
signal = truncate(Normal(5.3, 0.1), model_range...)
background = Uniform(model_range...)
model = MixtureModel([signal, background], [0.6, 0.4])
sW = SPlot(model)

x = rand(signal, 600) ⋃ rand(background, 400)
w = sweights(model, x)

# access signal component weights
w_signal = view(w, 1, :)
```

## Questions for Clarification
Please expand on the above questions to guide the implementation.
