# Filter by source, target, and edge_type_id
src = lgn_edge_df[lgn_edge_df["source_" + pop1] == 0]
src_tgt = src[src["target_" + pop2] == 11]
src_tgt_id = src_tgt[src_tgt["edge_type_id"] == edge_type_id]
src_tgt_id_nsyns = src_tgt_id[src_tgt_id["nsyns"] == nsyn]
