"""Cost-model imbalance and halo traffic (k = 4, global_hierarchical_graph,
hidden-256 cost model) and display names."""
# Cost model, k = 4, global_hierarchical_graph, hidden-256 cost model
IMB = {"lon_bands": 1.025, "geographic": 1.028, "nils_weighted_chunks": 1.033,
       "nils_weighted_metis": 1.067, "nils_weighted_rcb": 1.077,
       "nils_weighted_mtkahypar_km1": 1.090,
       "test_noise01": 1.024, "test_noise03": 1.027, "test_noise10": 1.021,
       "test_noise30": 1.014, "test_skew10": 1.102, "test_skew20": 1.201,
       "test_skew10_noise03": 1.097, "test_random": 1.029}
HALO = {"lon_bands": 2090, "geographic": 3561, "nils_weighted_chunks": 69234,
        "nils_weighted_metis": 1070, "nils_weighted_rcb": 1845,
        "nils_weighted_mtkahypar_km1": 945,
        "test_noise01": 3414, "test_noise03": 6780, "test_noise10": 14384,
        "test_noise30": 33490, "test_skew10": 2232, "test_skew20": 2296,
        "test_skew10_noise03": 6349, "test_random": 64218}
NAMES = {"lon_bands": "Longitude bands", "geographic": "Per-layer split",
         "nils_weighted_chunks": "Node-ID chunks",
         "nils_weighted_metis": "METIS on G0",
         "nils_weighted_rcb": "RCB on G0",
         "nils_weighted_mtkahypar_km1": "Mt-KaHyPar on G0",
         "test_noise01": "noise 1%", "test_noise03": "noise 3%",
         "test_noise10": "noise 10%", "test_noise30": "noise 30%",
         "test_skew10": "skew 10%", "test_skew20": "skew 20%",
         "test_skew10_noise03": "skew 10% + noise 3%",
         "test_random": "random"}
