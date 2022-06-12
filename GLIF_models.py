from pygenn.genn_model import GeNNModel, create_custom_current_source_class, create_custom_neuron_class

GLIF1 = create_custom_neuron_class(
    "GLIF1",
    param_names = ["C", "G", "El", "V_thres"],
    var_name_types=[("V", "scalar")],
    sim_code='$(V)+=1/$(C)*($(Isyn)-$(G)*($(V)-$(El)))*DT;',
    threshold_condition_code='$(V)>=$(V_thres)',
    reset_code = '$(V)=$(El);',
)