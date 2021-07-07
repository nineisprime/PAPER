

"""
tree_tools.py:
    Main file
    mcmcInfer: root-and-graft walk with fixed K >= 1
    gibbsSampling: Gibbs sampler with fixed K >= 1
    gibbsSamplingDP: Gibbs sampler with DP on K
    estimateAlphaEM: estimates alpha in PA tree, assumes beta=1
    


basic_converge_test.py:
    Testing convergence of mcmcInfer function
    
basic_converge_test2.py:
    Test mcmcInferToConvergence function
    
code_test.py:
    Test gibbsTree function

estimate_alpha_test.py:
    Test estimateAlphaEM

degree_test.py:
    Simulate degree distribution for theory

visualize_karate_community.py:
    Runs mcmcInfer on karate club network 
    creates figure for NSF grant proposal
    
present_visualize_PA.py
present_visualize_UA.py
    
    
process_karate_net.py
process_flu_net.py
process_scimath_net.py
process_real_graph.py




infect_tools.py:    



"""



"""
Data:
    
    Flu network:
        graf = igraph.read("data/flu_net.gml")

    Karate network:
        graf = igraph.read("data/karate.gml")
        
    Political blog network:
        graf = Graph.Read_GML("data/polblogs.gml")
        
    ca-MathSciNet:
        graf = Graph.Read_Ncol("ca-MathSciNet.mtx")


    
    
"""