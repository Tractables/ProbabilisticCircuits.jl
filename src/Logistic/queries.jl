# export class_conditional_likelihood_per_instance
# # Class Conditional Probability
# function class_conditional_likelihood_per_instance(fc::FlowΔ, 
#                                                     classes::Int, 
#                                                     batch::PlainXData{Bool})
#     lc = origin(origin(fc))
#     @assert(lc isa LogisticΔ)
#     pass_up_down(fc, batch)
#     likelihoods = zeros(num_examples(batch), classes)
#     for n in fc
#         orig = logistic_origin(n)
#         if orig isa Logistic⋀Node
#             # For each class. orig.thetas is 2D so used eachcol
#             for (idx, thetaC) in enumerate(eachcol(orig.thetas))
#                 foreach(n.children, thetaC) do c, theta
#                     likelihoods[:, idx] .+= prod_fast(downflow(n), pr_factors(origin(c))) .* theta
#                 end
#             end
#         end
#     end
#     likelihoods
# end

# """
# Calculate conditional log likelihood for a batch of samples with evidence P(c | x).
# (Also returns the generated FlowΔ)
# """
# function class_conditional_likelihood_per_instance(lc::LogisticΔ, 
#                                                     classes::Int, 
#                                                     batch::PlainXData{Bool})
#     opts = (max_factors = 2, compact⋀=false, compact⋁=false)
#     fc = FlowΔ(lc, num_examples(batch), Float64, opts)
#     (fc, class_conditional_likelihood_per_instance(fc, classes, batch))
# end

